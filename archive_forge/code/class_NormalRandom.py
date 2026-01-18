import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class NormalRandom(RandomDistribution):
    """
    Normally distributed (Gaussian) random number.

    Specified with mean mu and standard deviation sigma.
    See the random module for further details.
    """
    mu = param.Number(default=0.0, doc='Mean value.')
    sigma = param.Number(default=1.0, bounds=(0.0, None), doc='Standard deviation.')

    def __call__(self):
        super().__call__()
        return self.random_generator.normalvariate(self.mu, self.sigma)