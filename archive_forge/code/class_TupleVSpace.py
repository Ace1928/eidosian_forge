import itertools
from future.utils import with_metaclass
from .util import subvals
from .extend import (Box, primitive, notrace_primitive, VSpace, vspace,
class TupleVSpace(SequenceVSpace):
    seq_type = tuple_