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
def _compute_eigenvalues(alpha, beta):
    """Computes all eigenvalues of a Hermitian tridiagonal matrix."""

    def _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, x):
        """Implements the Sturm sequence recurrence."""
        with ops.name_scope('sturm'):
            n = alpha.shape[0]
            zeros = array_ops.zeros(array_ops.shape(x), dtype=dtypes.int32)
            ones = array_ops.ones(array_ops.shape(x), dtype=dtypes.int32)

            def sturm_step0():
                q = alpha[0] - x
                count = array_ops.where(q < 0, ones, zeros)
                q = array_ops.where(math_ops.equal(alpha[0], x), alpha0_perturbation, q)
                return (q, count)

            def sturm_step(i, q, count):
                q = alpha[i] - beta_sq[i - 1] / q - x
                count = array_ops.where(q <= pivmin, count + 1, count)
                q = array_ops.where(q <= pivmin, math_ops.minimum(q, -pivmin), q)
                return (q, count)
            q, count = sturm_step0()
            blocksize = 16
            i = 1
            peel = (n - 1) % blocksize
            unroll_cnt = peel

            def unrolled_steps(start, q, count):
                for j in range(unroll_cnt):
                    q, count = sturm_step(start + j, q, count)
                return (start + unroll_cnt, q, count)
            i, q, count = unrolled_steps(i, q, count)
            unroll_cnt = blocksize
            cond = lambda i, q, count: math_ops.less(i, n)
            _, _, count = while_loop.while_loop(cond, unrolled_steps, [i, q, count], back_prop=False)
            return count
    with ops.name_scope('compute_eigenvalues'):
        if alpha.dtype.is_complex:
            alpha = math_ops.real(alpha)
            beta_sq = math_ops.real(math_ops.conj(beta) * beta)
            beta_abs = math_ops.sqrt(beta_sq)
        else:
            beta_sq = math_ops.square(beta)
            beta_abs = math_ops.abs(beta)
        finfo = np.finfo(alpha.dtype.as_numpy_dtype)
        off_diag_abs_row_sum = array_ops.concat([beta_abs[:1], beta_abs[:-1] + beta_abs[1:], beta_abs[-1:]], axis=0)
        lambda_est_max = math_ops.minimum(finfo.max, math_ops.reduce_max(alpha + off_diag_abs_row_sum))
        lambda_est_min = math_ops.maximum(finfo.min, math_ops.reduce_min(alpha - off_diag_abs_row_sum))
        t_norm = math_ops.maximum(math_ops.abs(lambda_est_min), math_ops.abs(lambda_est_max))
        one = np.ones([], dtype=alpha.dtype.as_numpy_dtype)
        safemin = np.maximum(one / finfo.max, (one + finfo.eps) * finfo.tiny)
        pivmin = safemin * math_ops.maximum(one, math_ops.reduce_max(beta_sq))
        alpha0_perturbation = math_ops.square(finfo.eps * beta_abs[0])
        abs_tol = finfo.eps * t_norm
        if tol:
            abs_tol = math_ops.maximum(tol, abs_tol)
        max_it = finfo.nmant + 1
        asserts = None
        if select == 'a':
            target_counts = math_ops.range(n)
        elif select == 'i':
            asserts = check_ops.assert_less_equal(select_range[0], select_range[1], message='Got empty index range in select_range.')
            target_counts = math_ops.range(select_range[0], select_range[1] + 1)
        elif select == 'v':
            asserts = check_ops.assert_less(select_range[0], select_range[1], message='Got empty interval in select_range.')
        else:
            raise ValueError("'select must have a value in {'a', 'i', 'v'}.")
        if asserts:
            with ops.control_dependencies([asserts]):
                alpha = array_ops.identity(alpha)
        fudge = 2.1
        norm_slack = math_ops.cast(n, alpha.dtype) * fudge * finfo.eps * t_norm
        if select in {'a', 'i'}:
            lower = lambda_est_min - norm_slack - 2 * fudge * pivmin
            upper = lambda_est_max + norm_slack + fudge * pivmin
        else:
            lower = select_range[0] - norm_slack - 2 * fudge * pivmin
            upper = select_range[1] + norm_slack + fudge * pivmin
            first = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, lower)
            last = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, upper)
            target_counts = math_ops.range(first, last)
        upper = math_ops.minimum(upper, finfo.max)
        lower = math_ops.maximum(lower, finfo.min)
        target_shape = array_ops.shape(target_counts)
        lower = array_ops.broadcast_to(lower, shape=target_shape)
        upper = array_ops.broadcast_to(upper, shape=target_shape)
        pivmin = array_ops.broadcast_to(pivmin, target_shape)
        alpha0_perturbation = array_ops.broadcast_to(alpha0_perturbation, target_shape)

        def midpoint(lower, upper):
            return 0.5 * lower + 0.5 * upper

        def continue_binary_search(i, lower, upper):
            return math_ops.logical_and(math_ops.less(i, max_it), math_ops.less(abs_tol, math_ops.reduce_max(upper - lower)))

        def binary_search_step(i, lower, upper):
            mid = midpoint(lower, upper)
            counts = _sturm(alpha, beta_sq, pivmin, alpha0_perturbation, mid)
            lower = array_ops.where(counts <= target_counts, mid, lower)
            upper = array_ops.where(counts > target_counts, mid, upper)
            return (i + 1, lower, upper)
        _, lower, upper = while_loop.while_loop(continue_binary_search, binary_search_step, [0, lower, upper])
        return midpoint(lower, upper)