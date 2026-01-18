from numba.core import types, errors, ir, sigutils, ir_utils
from numba.core.typing.typeof import typeof_impl
from numba.core.transforms import find_region_inout_vars
from numba.core.ir_utils import build_definitions
import numba
class WithContext(object):
    """A dummy object for use as contextmanager.
    This can be used as a contextmanager.
    """
    is_callable = False

    def __enter__(self):
        pass

    def __exit__(self, typ, val, tb):
        pass

    def mutate_with_body(self, func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra):
        """Mutate the *blocks* to implement this contextmanager.

        Parameters
        ----------
        func_ir : FunctionIR
        blocks : dict[ir.Block]
        blk_start, blk_end : int
            labels of the starting and ending block of the context-manager.
        body_block: sequence[int]
            A sequence of int's representing labels of the with-body
        dispatcher_factory : callable
            A callable that takes a `FunctionIR` and returns a `Dispatcher`.
        """
        raise NotImplementedError