from copy import copy
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Callable, Dict, Union, List, Optional, no_type_check
import numpy as np
from scipy.special import erf
from pyquil.quilatom import TemplateWaveform, _update_envelope, _complex_str, Expression, substitute
@waveform('boxcar_kernel')
class BoxcarAveragerKernel(TemplateWaveform):
    scale: Optional[float] = None
    ' An optional global scaling factor. '
    phase: Optional[float] = None
    ' An optional phase shift factor. '
    detuning: Optional[float] = None
    ' An optional frequency detuning factor. '

    def out(self) -> str:
        output = 'boxcar_kernel('
        output += ', '.join([f'duration: {self.duration}'] + _optional_field_strs(self))
        output += ')'
        return output

    def __str__(self) -> str:
        return self.out()

    def samples(self, rate: float) -> np.ndarray:
        n = self.num_samples(rate)
        iqs = np.full(n, 1.0 / n, dtype=np.complex128)
        return _update_envelope(iqs, rate, scale=self.scale, phase=self.phase, detuning=self.detuning)