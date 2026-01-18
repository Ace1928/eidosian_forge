import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class tuple(with_metaclass(TupleMeta, tuple_)):

    def __new__(cls, xs):
        return make_sequence(tuple_, *xs)