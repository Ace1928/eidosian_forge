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
@staticmethod
def _validate_parameter_shift(device):
    if isinstance(device, qml.devices.Device):
        return (qml.gradients.param_shift, {}, device)
    model = device.capabilities().get('model', None)
    if model in {'qubit', 'qutrit'}:
        return (qml.gradients.param_shift, {}, device)
    if model == 'cv':
        return (qml.gradients.param_shift_cv, {'dev': device}, device)
    raise qml.QuantumFunctionError(f"Device {device.short_name} uses an unknown model ('{model}') that does not support the parameter-shift rule.")