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
def _update_original_device(self):
    if self.device is not self._original_device:
        if not self._tape_cached:
            self._original_device._num_executions += 1
        if hasattr(self._original_device, '_pre_rotated_state'):
            self._original_device._pre_rotated_state = self.device._pre_rotated_state
        if hasattr(self._original_device, '_state'):
            self._original_device._state = self.device._state