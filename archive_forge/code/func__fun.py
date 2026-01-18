from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
@wraps(fun)
def _fun(*args):
    return maybe_cached_unary_fun(args)