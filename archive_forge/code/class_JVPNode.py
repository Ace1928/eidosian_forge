from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
class JVPNode(Node):
    __slots__ = ['g']

    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        parent_gs = [parent.g for parent in parents]
        try:
            jvpmaker = primitive_jvps[fun]
        except KeyError:
            name = getattr(fun, '__name__', fun)
            raise NotImplementedError('JVP of {} wrt argnums {} not defined'.format(name, parent_argnums))
        self.g = jvpmaker(parent_argnums, parent_gs, value, args, kwargs)

    def initialize_root(self, g):
        self.g = g