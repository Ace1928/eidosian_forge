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
class StaticGetItemConstraint(object):

    def __init__(self, target, value, index, index_var, loc):
        self.target = target
        self.value = value
        self.index = index
        if index_var is not None:
            self.fallback = IntrinsicCallConstraint(target, operator.getitem, (value, index_var), {}, None, loc)
        else:
            self.fallback = None
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of static-get-item at {0}', self.loc):
            typevars = typeinfer.typevars
            for ty in typevars[self.value.name].get():
                sig = typeinfer.context.resolve_static_getitem(value=ty, index=self.index)
                if sig is not None:
                    itemty = sig.return_type
                    typeinfer.add_type(self.target, itemty, loc=self.loc)
                elif self.fallback is not None:
                    self.fallback(typeinfer)

    def get_call_signature(self):
        return self.fallback and self.fallback.get_call_signature()