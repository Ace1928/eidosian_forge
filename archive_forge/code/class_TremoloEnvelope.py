import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
class TremoloEnvelope(_Envelope):
    """A tremolo envelope, for modulation amplitude.

    A tremolo envelope that modulates the amplitude of the
    waveform with a sinusoidal pattern. The depth and rate
    of modulation can be specified. Depth is calculated as
    a percentage of the maximum amplitude. For example:
    a depth of 0.2 and amplitude of 0.5 will fluctuate
    the amplitude between 0.4 an 0.5.

    :Parameters:
        `depth` : float
            The amount of fluctuation, from 0.0 to 1.0.
        `rate` : float
            The fluctuation frequency, in seconds.
        `amplitude` : float
            The peak amplitude (volume), from 0.0 to 1.0.
    """

    def __init__(self, depth, rate, amplitude=0.5):
        self.depth = max(min(1.0, depth), 0)
        self.rate = rate
        self.amplitude = max(min(1.0, amplitude), 0)

    def get_generator(self, sample_rate, duration):
        total_bytes = int(sample_rate * duration)
        period = total_bytes / duration
        max_amplitude = self.amplitude
        min_amplitude = max(0.0, (1.0 - self.depth) * self.amplitude)
        step = _math.pi * 2 / period / self.rate
        for i in range(total_bytes):
            value = _math.sin(step * i)
            yield (value * (max_amplitude - min_amplitude) + min_amplitude)
        while True:
            yield 0