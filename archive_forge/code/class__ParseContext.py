from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class _ParseContext(object):

    def __init__(self, unary_ops, binary_ops, atomic_types, trace):
        self.op_stack = []
        self.noun_stack = []
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.atomic_types = atomic_types
        self.trace = trace
    __getstate__ = no_pickling