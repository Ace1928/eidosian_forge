from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('gaussian')
class GaussianWaveform(TemplateWaveform):
    """A Gaussian pulse."""
    fwhm: float
    ' The Full-Width-Half-Max of the Gaussian (seconds). '
    t0: float
    ' The center time coordinate of the Gaussian (seconds). '
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'gaussian('
        output += ', '.join([f'duration: {self.duration}', f'fwhm: {self.fwhm}', f't0: {self.t0}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        iqs = np.exp(-0.5 * (ts - self.t0) ** 2 / sigma ** 2)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)