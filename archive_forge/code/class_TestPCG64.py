import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
class TestPCG64(RNG):

    @classmethod
    def setup_class(cls):
        cls.bit_generator = PCG64
        cls.advance = 2 ** 63 + 2 ** 31 + 2 ** 15 + 1
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 64
        cls._extra_setup()