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
class IntrinsicCallConstraint(CallConstraint):

    def __call__(self, typeinfer):
        with new_error_context('typing of intrinsic-call at {0}', self.loc):
            fnty = self.func
            if fnty in utils.OPERATORS_TO_BUILTINS:
                fnty = typeinfer.resolve_value_type(None, fnty)
            self.resolve(typeinfer, typeinfer.typevars, fnty=fnty)