import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
def _hash_and_seed(self):
    """
        To be called between blocks of random number generation. A
        'block' can be an unbounded sequence of random numbers so long
        as the time value (as returned by time_fn) is guaranteed not
        to change within the block. If this condition holds, each
        block of random numbers is time-dependent.

        Note: param.random_seed is assumed to be integer or rational.
        """
    hashval = self._hashfn(self.time_fn(), param.random_seed)
    self.random_generator.seed(hashval)