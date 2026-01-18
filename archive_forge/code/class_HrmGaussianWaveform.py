from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('hrm_gaussian')
class HrmGaussianWaveform(TemplateWaveform):
    """A Hermite Gaussian waveform.

    REFERENCE: Effects of arbitrary laser or NMR pulse shapes on population
        inversion and coherence Warren S. Warren. 81, (1984); doi:
        10.1063/1.447644
    """
    fwhm: float
    ' The Full-Width-Half-Max of the Gaussian (seconds). '
    t0: float
    ' The center time coordinate of the Gaussian (seconds). '
    anh: float
    ' The anharmonicity of the qubit, f01-f12 (Hertz). '
    alpha: float
    ' Dimensionles DRAG parameter. '
    second_order_hrm_coeff: float
    ' Second order coefficient (see Warren 1984). '
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'hrm_gaussian('
        output += ', '.join([f'duration: {self.duration}', f'fwhm: {self.fwhm}', f't0: {self.t0}', f'anh: {self.anh}', f'alpha: {self.alpha}', f'second_order_hrm_coeff: {self.second_order_hrm_coeff}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        ts = np.arange(self.num_samples(rate), dtype=np.complex128) / rate
        sigma = 0.5 * self.fwhm / np.sqrt(2.0 * np.log(2.0))
        exponent_of_t = 0.5 * (ts - self.t0) ** 2 / sigma ** 2
        gauss = np.exp(-exponent_of_t)
        env = (1 - self.second_order_hrm_coeff * exponent_of_t) * gauss
        deriv_prefactor = -self.alpha / (2 * np.pi * self.anh)
        env_der = deriv_prefactor * (ts - self.t0) / sigma ** 2 * gauss * (self.second_order_hrm_coeff * (exponent_of_t - 1) - 1)
        iqs = env + 1j * env_der
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)