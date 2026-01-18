import pickle
from functools import partial
import numpy as np
import pytest
from numpy.testing import assert_equal, assert_, assert_array_equal
from numpy.random import (Generator, MT19937, PCG64, PCG64DXSM, Philox, SFC64)
class TestSFC64(RNG):

    @classmethod
    def setup_class(cls):
        cls.bit_generator = SFC64
        cls.advance = None
        cls.seed = [12345]
        cls.rg = Generator(cls.bit_generator(*cls.seed))
        cls.initial_state = cls.rg.bit_generator.state
        cls.seed_vector_bits = 192
        cls._extra_setup()