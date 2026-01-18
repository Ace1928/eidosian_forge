from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
@dataclass
class TemplateWaveform(QuilAtom):
    duration: float
    ' The duration [seconds] of the waveform. '

    def num_samples(self, rate: float) -> int:
        """The number of samples in the reference implementation of the waveform.

        Note: this does not include any hardware-enforced alignment (cf.
        documentation for `samples`).

        :param rate: The sample rate, in Hz.
        :return: The number of samples.

        """
        return int(np.ceil(self.duration * rate))

    def samples(self, rate: float) -> np.ndarray:
        """A reference implementation of waveform sample generation.

        Note: this is close but not always exactly equivalent to the actual IQ
        values produced by the waveform generators on Rigetti hardware. The
        actual ADC process imposes some alignment constraints on the waveform
        duration (in particular, it must be compatible with the clock rate).

        :param rate: The sample rate, in Hz.
        :returns: An array of complex samples.

        """
        raise NotImplementedError()