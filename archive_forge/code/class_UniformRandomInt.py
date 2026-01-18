import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class UniformRandomInt(RandomDistribution):
    """
    Specified with lbound and ubound; when called, return a random
    number in the inclusive range [lbound, ubound].

    See the randint function in the random module for further details.
    """
    lbound = param.Number(default=0, doc='Inclusive lower bound.')
    ubound = param.Number(default=1000, doc='Inclusive upper bound.')

    def __call__(self):
        super().__call__()
        x = self.random_generator.randint(self.lbound, self.ubound)
        return x