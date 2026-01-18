from __future__ import annotations
import functools
import warnings
from collections.abc import Mapping, Callable
from copy import deepcopy
from typing import Any
import numpy as np
import symengine as sym
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform
class _PulseType(type):
    """Metaclass to warn at isinstance check."""

    def __instancecheck__(cls, instance):
        cls_alias = getattr(cls, 'alias', None)
        warnings.warn(f"Typechecking with the symbolic pulse subclass will be deprecated. '{cls_alias}' subclass instance is turned into SymbolicPulse instance. Use self.pulse_type == '{cls_alias}' instead.", PendingDeprecationWarning)
        if not isinstance(instance, SymbolicPulse):
            return False
        return instance.pulse_type == cls_alias

    def __getattr__(cls, item):
        return NotImplemented