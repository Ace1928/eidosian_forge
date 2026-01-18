from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
class ParseNode(object):

    def __init__(self, type, token, args, origin):
        self.type = type
        self.token = token
        self.args = args
        self.origin = origin
    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        return repr_pretty_impl(p, self, [self.type, self.token, self.args])
    __getstate__ = no_pickling