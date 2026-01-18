from .mtrand import RandomState
from ._philox import Philox
from ._pcg64 import PCG64, PCG64DXSM
from ._sfc64 import SFC64
from ._generator import Generator
from ._mt19937 import MT19937
def __randomstate_ctor(bit_generator_name='MT19937', bit_generator_ctor=__bit_generator_ctor):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name : str
        String containing the core BitGenerator's name
    bit_generator_ctor : callable, optional
        Callable function that takes bit_generator_name as its only argument
        and returns an instantized bit generator.

    Returns
    -------
    rs : RandomState
        Legacy RandomState using the named core BitGenerator
    """
    return RandomState(bit_generator_ctor(bit_generator_name))