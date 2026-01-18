from numba import typeof
from numba.core import types
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba.np.ufunc.sigparse import parse_signature
from numba.np.numpy_support import ufunc_find_matching_loop
from numba.core import serialize
import functools
def _num_args_match(self, *args):
    parsed_sig = parse_signature(self.gufunc_builder.signature)
    return len(args) == len(parsed_sig[0]) + len(parsed_sig[1])