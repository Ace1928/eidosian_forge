import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
class DelItemConstraint(object):

    def __init__(self, target, index, loc):
        self.target = target
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of delitem at {0}', self.loc):
            typevars = typeinfer.typevars
            if not all((typevars[var.name].defined for var in (self.target, self.index))):
                return
            targetty = typevars[self.target.name].getone()
            idxty = typevars[self.index.name].getone()
            sig = typeinfer.context.resolve_delitem(targetty, idxty)
            if sig is None:
                raise TypingError('Cannot resolve delitem: %s[%s]' % (targetty, idxty), loc=self.loc)
            self.signature = sig

    def get_call_signature(self):
        return self.signature