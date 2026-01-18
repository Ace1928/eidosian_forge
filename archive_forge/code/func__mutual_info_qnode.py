from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
@mutual_info.custom_qnode_transform
def _mutual_info_qnode(self, qnode, targs, tkwargs):
    if tkwargs.get('device', False):
        raise ValueError("Cannot provide a 'device' value directly to the mutual_info decorator when transforming a QNode.")
    if tkwargs.get('device_wires', None):
        raise ValueError("Cannot provide a 'device_wires' value directly to the mutual_info decorator when transforming a QNode.")
    tkwargs.setdefault('device', qnode.device)
    tkwargs.setdefault('device_wires', qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)