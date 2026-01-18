import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class UniformRandom(RandomDistribution):
    """
    Specified with lbound and ubound; when called, return a random
    number in the range [lbound, ubound).

    See the random module for further details.
    """
    lbound = param.Number(default=0.0, doc='Inclusive lower bound.')
    ubound = param.Number(default=1.0, doc='Exclusive upper bound.')

    def __call__(self):
        super().__call__()
        return self.random_generator.uniform(self.lbound, self.ubound)