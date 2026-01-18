from . import _pickle
from . import _common
from . import _bounded_integers
from ._generator import Generator, default_rng
from .bit_generator import SeedSequence, BitGenerator
from ._mt19937 import MT19937
from ._pcg64 import PCG64, PCG64DXSM
from ._philox import Philox
from ._sfc64 import SFC64
from .mtrand import *
from numpy._pytesttester import PytestTester
def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.

    Note that the state of the RandomState returned here is irrelevant, as this
    function's entire purpose is to return a newly allocated RandomState whose
    state pickle can set.  Consequently the RandomState returned by this function
    is a freshly allocated copy with a seed=0.

    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion

    """
    return RandomState(seed=0)