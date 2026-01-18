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
def _update_gradient_fn(self, shots=None):
    if self.diff_method is None:
        self._interface = None
        self.gradient_fn = None
        self.gradient_kwargs = {}
        return
    if self.interface == 'auto' and self.diff_method in ['backprop', 'best']:
        if self.diff_method == 'backprop':
            if isinstance(self.device, Device):
                backprop_devices = self.device.capabilities().get('passthru_devices', None)
                if backprop_devices is None:
                    raise qml.QuantumFunctionError(f'The {self.device.short_name} device does not support native computations with autodifferentiation frameworks.')
        return
    self.gradient_fn, self.gradient_kwargs, self.device = self.get_gradient_fn(self._original_device, self.interface, self.diff_method, shots=shots)
    self.gradient_kwargs.update(self._user_gradient_kwargs or {})