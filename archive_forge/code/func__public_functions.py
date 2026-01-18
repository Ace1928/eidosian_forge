import inspect
import itertools
from . import _funcs_impl, _reductions_impl
from ._normalizations import normalizer
def _public_functions(mod):

    def is_public_function(f):
        return inspect.isfunction(f) and (not f.__name__.startswith('_'))
    return inspect.getmembers(mod, is_public_function)