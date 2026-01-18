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
def Sawtooth(duration: int | ParameterExpression, amp: float | ParameterExpression, phase: float | ParameterExpression, freq: float | ParameterExpression | None=None, angle: float | ParameterExpression | None=0.0, name: str | None=None, limit_amplitude: bool | None=None) -> ScalableSymbolicPulse:
    """A sawtooth pulse.

    The envelope of the pulse is given by:

    .. math::

        f(x) = 2\\text{A}\\left[g\\left(x\\right)-
            \\lfloor g\\left(x\\right)+\\frac{1}{2}\\rfloor\\right]

    where :math:`\\text{A} = \\text{amp} \\times\\exp\\left(i\\times\\text{angle}\\right)`,
    :math:`g\\left(x\\right)=x\\times\\text{freq}+\\frac{\\text{phase}}{2\\pi}`,
    and :math:`\\lfloor ...\\rfloor` is the floor operation.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the sawtooth wave. Wave range is [-`amp`,`amp`].
        phase: The phase of the sawtooth wave (note that this is not equivalent to the angle
            of the complex amplitude)
        freq: The frequency of the sawtooth wave, in terms of 1 over sampling period.
            If not provided defaults to a single cycle (i.e :math:'\\frac{1}{\\text{duration}}').
            The frequency is limited to the range :math:`\\left(0,0.5\\right]` (the Nyquist frequency).
        angle: The angle in radians of the complex phase factor uniformly
            scaling the pulse. Default value 0.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.
    """
    if freq is None:
        freq = 1 / duration
    parameters = {'freq': freq, 'phase': phase}
    _t, _duration, _amp, _angle, _freq, _phase = sym.symbols('t, duration, amp, angle, freq, phase')
    lin_expr = _t * _freq + _phase / (2 * sym.pi)
    envelope_expr = 2 * _amp * sym.exp(sym.I * _angle) * (lin_expr - sym.floor(lin_expr + 1 / 2))
    consts_expr = sym.And(_freq > 0, _freq < 0.5)
    valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
    return ScalableSymbolicPulse(pulse_type='Sawtooth', duration=duration, amp=amp, angle=angle, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope_expr, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)