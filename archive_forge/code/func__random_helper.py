from ..base import numeric_types, _Null
from ..context import current_context
from . import _internal
from .ndarray import NDArray
def _random_helper(random, sampler, params, shape, dtype, ctx, out, kwargs):
    """Helper function for random generators."""
    if isinstance(params[0], NDArray):
        for i in params[1:]:
            assert isinstance(i, NDArray), 'Distribution parameters must all have the same type, but got both %s and %s.' % (type(params[0]), type(i))
        return sampler(*params, shape=shape, dtype=dtype, out=out, **kwargs)
    elif isinstance(params[0], numeric_types):
        if ctx is None:
            ctx = current_context()
        if shape is _Null and out is None:
            shape = 1
        for i in params[1:]:
            assert isinstance(i, numeric_types), 'Distribution parameters must all have the same type, but got both %s and %s.' % (type(params[0]), type(i))
        return random(*params, shape=shape, dtype=dtype, ctx=ctx, out=out, **kwargs)
    raise ValueError('Distribution parameters must be either NDArray or numbers, but got %s.' % type(params[0]))