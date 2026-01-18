from typing import Sequence, Callable
import functools
from functools import partial
import warnings
import numpy as np
import pennylane as qml
from pennylane.circuit_graph import LayerData
from pennylane.queuing import WrappedObj
from pennylane.transforms import transform
def _contract_metric_tensor_with_cjac(mt, cjac, tape):
    """Execute the contraction of pre-computed classical Jacobian(s)
    and the metric tensor of a tape in order to obtain the hybrid
    metric tensor of a QNode.

    Args:
        mt (array): Metric tensor of a tape (2-dimensional)
        cjac (array or tuple[array]): The classical Jacobian of a QNode

    Returns:
        array or tuple[array]: Hybrid metric tensor(s) of the QNode.
        The number of metric tensors depends on the number of QNode arguments
        for which the classical Jacobian was computed, the tensor shape(s)
        depend on the shape of these QNode arguments.
    """
    if isinstance(mt, tuple) and len(mt) == 1:
        mt = mt[0]
    if isinstance(cjac, tuple):
        metric_tensors = tuple((qml.math.tensordot(c, qml.math.tensordot(mt, c, axes=[[-1], [0]]), axes=[[0], [0]]) for c in cjac if c is not None))
        return metric_tensors[0] if len(metric_tensors) == 1 else metric_tensors
    is_square = cjac.shape == (1,) or (cjac.ndim == 2 and cjac.shape[0] == cjac.shape[1])
    if is_square and qml.math.allclose(cjac, qml.numpy.eye(cjac.shape[0])):
        return mt
    mt_cjac = qml.math.tensordot(mt, cjac, axes=[[-1], [0]])
    mt = qml.math.tensordot(cjac, mt_cjac, axes=[[0], [0]])
    return mt