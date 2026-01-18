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
def Sech(duration: int | ParameterValueType, amp: float | ParameterExpression, sigma: float | ParameterExpression, angle: float | ParameterExpression | None=0.0, name: str | None=None, zero_ends: bool | None=True, limit_amplitude: bool | None=None) -> ScalableSymbolicPulse:
    """An unnormalized sech pulse.

    The sech function is centered around the halfway point of the pulse,
    and the envelope of the pulse is given by:

    .. math::

        f(x) = \\text{A}\\text{sech}\\left(
            \\frac{x-\\mu}{\\text{sigma}}\\right)  ,  0 <= x < duration

    where :math:`\\text{A} = \\text{amp} \\times\\exp\\left(i\\times\\text{angle}\\right)`,
    and :math:`\\mu=\\text{duration}/2`.

    If `zero_ends` is set to `True`, the output `y` is modified:
    .. math::

        y\\left(x\\right) \\mapsto \\text{A}\\frac{y-y^{*}}{\\text{A}-y^{*}},

    where :math:`y^{*}` is the value of :math:`y` at the endpoints (at :math:`x=-1
    and :math:`x=\\text{duration}+1`). This shifts the endpoints value to zero, while also
    rescaling to preserve the amplitude at `:math:`\\text{duration}/2``.

    Args:
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the pulse (the value at the midpoint `duration`/2).
        sigma: A measure of how wide or narrow the sech peak is in terms of `dt`;
            described mathematically in the class docstring.
        angle: The angle in radians of the complex phase factor uniformly
            scaling the pulse. Default value 0.
        name: Display name for this pulse envelope.
        zero_ends: If True, zeros the ends at x = -1, x = `duration` + 1,
            but rescales to preserve `amp`. Default value True.
        limit_amplitude: If ``True``, then limit the amplitude of the
            waveform to 1. The default is ``True`` and the amplitude is constrained to 1.

    Returns:
        ScalableSymbolicPulse instance.
    """
    parameters = {'sigma': sigma}
    _t, _duration, _amp, _angle, _sigma = sym.symbols('t, duration, amp, angle, sigma')
    complex_amp = _amp * sym.exp(sym.I * _angle)
    envelope_expr = complex_amp * sym.sech((_t - _duration / 2) / _sigma)
    if zero_ends:
        shift_val = complex_amp * sym.sech((-1 - _duration / 2) / _sigma)
        envelope_expr = complex_amp * (envelope_expr - shift_val) / (complex_amp - shift_val)
    consts_expr = _sigma > 0
    valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
    return ScalableSymbolicPulse(pulse_type='Sech', duration=duration, amp=amp, angle=angle, parameters=parameters, name=name, limit_amplitude=limit_amplitude, envelope=envelope_expr, constraints=consts_expr, valid_amp_conditions=valid_amp_conditions_expr)