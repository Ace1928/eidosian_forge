import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class UniformRandomOffset(RandomDistribution):
    """
    Identical to UniformRandom, but specified by mean and range.
    When called, return a random number in the range
    [mean - range/2, mean + range/2).

    See the random module for further details.
    """
    mean = param.Number(default=0.0, doc='Mean value')
    range = param.Number(default=1.0, bounds=(0.0, None), doc='\n        Difference of maximum and minimum value')

    def __call__(self):
        super().__call__()
        return self.random_generator.uniform(self.mean - self.range / 2.0, self.mean + self.range / 2.0)