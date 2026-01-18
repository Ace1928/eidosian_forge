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
class ScalableSymbolicPulse(SymbolicPulse):
    """Subclass of :class:`SymbolicPulse` for pulses with scalable envelope.

    Instance of :class:`ScalableSymbolicPulse` behaves the same as an instance of
    :class:`SymbolicPulse`, but its envelope is assumed to have a scalable form
    :math:`\\text{amp}\\times\\exp\\left(i\\times\\text{angle}\\right)\\times\\text{F}
    \\left(t,\\text{parameters}\\right)`,
    where :math:`\\text{F}` is some function describing the rest of the envelope,
    and both `amp` and `angle` are real (float). Note that both `amp` and `angle` are
    stored in the :attr:`parameters` dictionary of the :class:`ScalableSymbolicPulse`
    instance.

    When two :class:`ScalableSymbolicPulse` objects are equated, instead of comparing
    `amp` and `angle` individually, only the complex amplitude
    :math:'\\text{amp}\\times\\exp\\left(i\\times\\text{angle}\\right)' is compared.
    """

    def __init__(self, pulse_type: str, duration: ParameterExpression | int, amp: ParameterValueType, angle: ParameterValueType, parameters: dict[str, ParameterExpression | complex] | None=None, name: str | None=None, limit_amplitude: bool | None=None, envelope: sym.Expr | None=None, constraints: sym.Expr | None=None, valid_amp_conditions: sym.Expr | None=None):
        """Create a scalable symbolic pulse.

        Args:
            pulse_type: Display name of this pulse shape.
            duration: Duration of pulse.
            amp: The magnitude of the complex amplitude of the pulse.
            angle: The phase of the complex amplitude of the pulse.
            parameters: Dictionary of pulse parameters that defines the pulse envelope.
            name: Display name for this particular pulse envelope.
            limit_amplitude: If ``True``, then limit the absolute value of the amplitude of the
                waveform to 1. The default is ``True`` and the amplitude is constrained to 1.
            envelope: Pulse envelope expression.
            constraints: Pulse parameter constraint expression.
            valid_amp_conditions: Extra conditions to skip a full-waveform check for the
                amplitude limit. If this condition is not met, then the validation routine
                will investigate the full-waveform and raise an error when the amplitude norm
                of any data point exceeds 1.0. If not provided, the validation always
                creates a full-waveform.

        Raises:
            PulseError: If ``amp`` is complex.
        """
        if isinstance(amp, complex):
            raise PulseError("amp represents the magnitude of the complex amplitude and can't be complex")
        if not isinstance(parameters, dict):
            parameters = {'amp': amp, 'angle': angle}
        else:
            parameters = deepcopy(parameters)
            parameters['amp'] = amp
            parameters['angle'] = angle
        super().__init__(pulse_type=pulse_type, duration=duration, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope, constraints=constraints, valid_amp_conditions=valid_amp_conditions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalableSymbolicPulse):
            return NotImplemented
        if self._pulse_type != other._pulse_type:
            return False
        if self._envelope != other._envelope:
            return False
        complex_amp1 = self.amp * np.exp(1j * self.angle)
        complex_amp2 = other.amp * np.exp(1j * other.angle)
        if isinstance(complex_amp1, ParameterExpression) or isinstance(complex_amp2, ParameterExpression):
            if complex_amp1 != complex_amp2:
                return False
        elif not np.isclose(complex_amp1, complex_amp2):
            return False
        for key in self.parameters:
            if key not in ['amp', 'angle'] and self.parameters[key] != other.parameters[key]:
                return False
        return True