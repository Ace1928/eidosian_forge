from itertools import repeat
from autograd.wrap_util import wraps
from autograd.util import subvals, toposort
from autograd.tracer import trace, Node
from functools import partial
class ConstGraphNode(Node):
    __slots__ = ['parents', 'partial_fun']

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        args = subvals(args, zip(parent_argnums, repeat(None)))

        def partial_fun(partial_args):
            return fun(*subvals(args, zip(parent_argnums, partial_args)), **kwargs)
        self.parents = parents
        self.partial_fun = partial_fun

    def initialize_root(self):
        self.parents = []