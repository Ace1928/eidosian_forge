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
class TypedGetItemConstraint(object):

    def __init__(self, target, value, dtype, index, loc):
        self.target = target
        self.value = value
        self.dtype = dtype
        self.index = index
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of typed-get-item at {0}', self.loc):
            typevars = typeinfer.typevars
            idx_ty = typevars[self.index.name].get()
            ty = typevars[self.value.name].get()
            self.signature = Signature(self.dtype, ty + idx_ty, None)
            typeinfer.add_type(self.target, self.dtype, loc=self.loc)

    def get_call_signature(self):
        return self.signature