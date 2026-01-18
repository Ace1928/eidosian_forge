from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
def _metric_tensor_hadamard(tape, argnum, allow_nonunitary, aux_wire, device_wires):
    """Generate the quantum tapes that execute the Hadamard tests
    to compute the first term of off block-diagonal metric entries
    and combine them with the covariance matrix-based block-diagonal tapes.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to find the metric tensor of
        argnum (list[int]): Trainable tape-parameter indices with respect to which the metric tensor
            is computed.
        allow_nonunitary (bool): Whether non-unitary operations are allowed in circuits
            created by the transform. Only relevant if ``approx`` is ``None``
            Should be set to ``True`` if possible to reduce cost.
        aux_wire (int or .wires.Wires): Auxiliary wire to be used for
            Hadamard tests. By default, a suitable wire is inferred from the number
            of used wires in the original circuit.
        device_wires (.wires.Wires): Wires of the device that is going to be used for the
            metric tensor. Facilitates finding a default for ``aux_wire`` if ``aux_wire``
            is ``None`` .

    Returns:
        list[pennylane.tape.QuantumTape]: Tapes to evaluate the metric tensor
        callable: processing function to obtain the metric tensor from the tape results
    """
    diag_tapes, diag_proc_fn, obs_list, coeffs_list, in_argnum_list, layer_ids, obs_ids = _metric_tensor_cov_matrix(tape, argnum, diag_approx=False)
    graph = tape.graph
    par_idx_to_trainable_idx = {idx: i for i, idx in enumerate(sorted(tape.trainable_params))}
    layers = []
    for layer, in_argnum in zip(graph.iterate_parametrized_layers(), in_argnum_list):
        if not any(in_argnum):
            continue
        pre_ops, ops, param_idx, post_ops = layer
        new_ops = []
        new_param_idx = []
        for o, idx, param_in_argnum in zip(ops, param_idx, in_argnum):
            if param_in_argnum:
                new_ops.append(o)
                new_param_idx.append(par_idx_to_trainable_idx[idx])
        layers.append(LayerData(pre_ops, new_ops, new_param_idx, post_ops))
    if len(layers) <= 1:
        return (diag_tapes, diag_proc_fn)
    aux_wire = _get_aux_wire(aux_wire, tape, device_wires)
    first_term_tapes = []
    ids = []
    block_sizes = []
    for idx_i, layer_i in enumerate(layers):
        block_sizes.append(len(layer_i.param_inds))
        for layer_j in layers[idx_i + 1:]:
            _tapes, _ids = _get_first_term_tapes(layer_i, layer_j, allow_nonunitary, aux_wire)
            first_term_tapes.extend(_tapes)
            ids.extend(_ids)
    tapes = diag_tapes + first_term_tapes
    blocks = []
    for in_argnum in in_argnum_list:
        d = len(in_argnum)
        blocks.append(qml.math.ones((d, d)))
    mask = 1 - qml.math.block_diag(blocks)
    num_diag_tapes = len(diag_tapes)

    def processing_fn(results):
        """Postprocessing function for the full metric tensor."""
        nonlocal mask
        diag_res, off_diag_res = (results[:num_diag_tapes], results[num_diag_tapes:])
        diag_mt = diag_proc_fn(diag_res)
        off_diag_res = [qml.math.expand_dims(res, 0) for res in off_diag_res]
        mask = qml.math.convert_like(mask, diag_mt)
        first_term = qml.math.zeros_like(diag_mt)
        if ids:
            off_diag_res = qml.math.stack(off_diag_res, 1)[0]
            inv_ids = [_id[::-1] for _id in ids]
            first_term = qml.math.scatter_element_add(first_term, list(zip(*ids)), off_diag_res)
            first_term = qml.math.scatter_element_add(first_term, list(zip(*inv_ids)), off_diag_res)
        expvals = qml.math.zeros_like(first_term[0])
        for i, (layer_i, obs_i) in enumerate(zip(layer_ids, obs_ids)):
            if layer_i is not None and obs_i is not None:
                prob = diag_res[layer_i]
                o = obs_list[layer_i][obs_i]
                l = qml.math.cast(o.eigvals(), dtype=np.float64)
                w = tape.wires.indices(o.wires)
                p = qml.math.marginal_prob(prob, w)
                expvals = qml.math.scatter_element_add(expvals, (i,), qml.math.dot(l, p))
        second_term = qml.math.tensordot(expvals, expvals, axes=0) * mask
        off_diag_mt = first_term - second_term
        coeffs_gen = (c for c in qml.math.hstack(coeffs_list))
        interface = qml.math.get_interface(*results)
        extended_coeffs_list = qml.math.asarray([next(coeffs_gen) if param_in_argnum else 0.0 for param_in_argnum in qml.math.hstack(in_argnum_list)], like=interface)
        scale = qml.math.tensordot(extended_coeffs_list, extended_coeffs_list, axes=0)
        off_diag_mt = scale * off_diag_mt
        mt = off_diag_mt + diag_mt
        return mt
    return (tapes, processing_fn)