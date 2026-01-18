from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class _StackOperator(object):

    def __init__(self, op, token):
        self.op = op
        self.token = token
    __getstate__ = no_pickling