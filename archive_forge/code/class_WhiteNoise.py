import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
class WhiteNoise(SynthesisSource):

    def __init__(self, duration, frequency=440, sample_rate=44800, envelope=None):
        """Create a random white noise waveform."""
        super().__init__(noise_generator(frequency, sample_rate), duration, sample_rate, envelope)