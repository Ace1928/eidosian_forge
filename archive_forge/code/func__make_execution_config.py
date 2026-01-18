import copy
import functools
import inspect
import warnings
from collections.abc import Sequence
from typing import Union
import logging
import pennylane as qml
from pennylane import Device
from pennylane.measurements import CountsMP, MidMeasureMP, Shots
from pennylane.tape import QuantumTape, QuantumScript
from .execution import INTERFACE_MAP, SUPPORTED_INTERFACES
from .set_shots import set_shots
def _make_execution_config(circuit: 'QNode') -> 'qml.devices.ExecutionConfig':
    if circuit.gradient_fn is None:
        _gradient_method = None
    elif isinstance(circuit.gradient_fn, str):
        _gradient_method = circuit.gradient_fn
    else:
        _gradient_method = 'gradient-transform'
    grad_on_execution = circuit.execute_kwargs.get('grad_on_execution')
    if circuit.interface == 'jax':
        grad_on_execution = False
    elif grad_on_execution == 'best':
        grad_on_execution = None
    return qml.devices.ExecutionConfig(interface=circuit.interface, gradient_method=_gradient_method, grad_on_execution=grad_on_execution, use_device_jacobian_product=circuit.execute_kwargs['device_vjp'])