from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
def _tf_jac(circ):
    """TF jacobian as a callable function"""
    import tensorflow as tf

    def wrapper(*args, **kwargs):
        with tf.GradientTape() as tape:
            loss = circ(*args, **kwargs)
        return tape.jacobian(loss, args)
    return wrapper