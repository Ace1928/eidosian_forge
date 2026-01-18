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
def _validate_device_method(device):
    if isinstance(device, Device):
        if device.capabilities().get('provides_jacobian', False):
            return ('device', {}, device)
        name = device.short_name
    else:
        config = qml.devices.ExecutionConfig(gradient_method='device')
        if device.supports_derivatives(config):
            return ('device', {}, device)
        name = device.name
    raise qml.QuantumFunctionError(f'The {name} device does not provide a native method for computing the jacobian.')