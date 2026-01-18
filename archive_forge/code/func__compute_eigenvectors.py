import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _compute_eigenvectors(alpha, beta, eigvals):
    """Implements inverse iteration to compute eigenvectors."""
    with ops.name_scope('compute_eigenvectors'):
        k = array_ops.size(eigvals)
        n = array_ops.size(alpha)
        alpha = math_ops.cast(alpha, dtype=beta.dtype)
        gap = eigvals[1:] - eigvals[:-1]
        eps = np.finfo(eigvals.dtype.as_numpy_dtype).eps
        t_norm = math_ops.maximum(math_ops.abs(eigvals[0]), math_ops.abs(eigvals[-1]))
        gaptol = np.sqrt(eps) * t_norm
        close = math_ops.less(gap, gaptol)
        left_neighbor_close = array_ops.concat([[False], close], axis=0)
        right_neighbor_close = array_ops.concat([close, [False]], axis=0)
        ortho_interval_start = math_ops.logical_and(math_ops.logical_not(left_neighbor_close), right_neighbor_close)
        ortho_interval_start = array_ops.squeeze(array_ops.where_v2(ortho_interval_start), axis=-1)
        ortho_interval_end = math_ops.logical_and(left_neighbor_close, math_ops.logical_not(right_neighbor_close))
        ortho_interval_end = array_ops.squeeze(array_ops.where_v2(ortho_interval_end), axis=-1) + 1
        num_clusters = array_ops.size(ortho_interval_end)
        v0 = math_ops.cast(stateless_random_ops.stateless_random_normal(shape=(k, n), seed=[7, 42]), dtype=beta.dtype)
        nrm_v = norm(v0, axis=1)
        v0 = v0 / nrm_v[:, array_ops.newaxis]
        zero_nrm = constant_op.constant(0, shape=nrm_v.shape, dtype=nrm_v.dtype)
        eigvals_cast = math_ops.cast(eigvals, dtype=beta.dtype)
        alpha_shifted = alpha[array_ops.newaxis, :] - eigvals_cast[:, array_ops.newaxis]
        beta = array_ops.tile(beta[array_ops.newaxis, :], [k, 1])
        diags = [beta, alpha_shifted, math_ops.conj(beta)]

        def orthogonalize_close_eigenvectors(eigenvectors):

            def orthogonalize_cluster(cluster_idx, eigenvectors):
                start = ortho_interval_start[cluster_idx]
                end = ortho_interval_end[cluster_idx]
                update_indices = array_ops.expand_dims(math_ops.range(start, end), -1)
                vectors_in_cluster = eigenvectors[start:end, :]
                q, _ = qr(transpose(vectors_in_cluster))
                vectors_to_update = transpose(q)
                eigenvectors = array_ops.tensor_scatter_nd_update(eigenvectors, update_indices, vectors_to_update)
                return (cluster_idx + 1, eigenvectors)
            _, eigenvectors = while_loop.while_loop(lambda i, ev: math_ops.less(i, num_clusters), orthogonalize_cluster, [0, eigenvectors])
            return eigenvectors

        def continue_iteration(i, _, nrm_v, nrm_v_old):
            max_it = 5
            min_norm_growth = 0.1
            norm_growth_factor = constant_op.constant(1 + min_norm_growth, dtype=nrm_v.dtype)
            return math_ops.logical_and(math_ops.less(i, max_it), math_ops.reduce_any(math_ops.greater_equal(math_ops.real(nrm_v), math_ops.real(norm_growth_factor * nrm_v_old))))

        def inverse_iteration_step(i, v, nrm_v, nrm_v_old):
            v = tridiagonal_solve(diags, v, diagonals_format='sequence', partial_pivoting=True, perturb_singular=True)
            nrm_v_old = nrm_v
            nrm_v = norm(v, axis=1)
            v = v / nrm_v[:, array_ops.newaxis]
            v = orthogonalize_close_eigenvectors(v)
            return (i + 1, v, nrm_v, nrm_v_old)
        _, v, nrm_v, _ = while_loop.while_loop(continue_iteration, inverse_iteration_step, [0, v0, nrm_v, zero_nrm])
        return transpose(v)