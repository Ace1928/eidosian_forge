from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('erf_square')
class ErfSquareWaveform(TemplateWaveform):
    """A pulse with a flat top and edges that are error functions (erf)."""
    risetime: float
    ' The width of each of the rise and fall sections of the pulse (seconds). '
    pad_left: float
    ' Amount of zero-padding to add to the left of the pulse (seconds).'
    pad_right: float
    ' Amount of zero-padding to add to the right of the pulse (seconds). '
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'erf_square('
        output += ', '.join([f'duration: {self.duration}', f'risetime: {self.risetime}', f'pad_left: {self.pad_left}', f'pad_right: {self.pad_right}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        fwhm = 0.5 * self.risetime
        t1 = fwhm
        t2 = self.duration - fwhm
        sigma = 0.5 * fwhm / np.sqrt(2.0 * np.log(2.0))
        vals = 0.5 * (erf((ts - t1) / sigma) - erf((ts - t2) / sigma))
        zeros_left = np.zeros(int(np.ceil(self.pad_left * rate)), dtype=np.complex128)
        zeros_right = np.zeros(int(np.ceil(self.pad_right * rate)), dtype=np.complex128)
        iqs = np.concatenate((zeros_left, vals, zeros_right))
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)