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
class Propagate(object):
    """
    A simple constraint for direct propagation of types for assignments.
    """

    def __init__(self, dst, src, loc):
        self.dst = dst
        self.src = src
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of assignment at {0}', self.loc, loc=self.loc):
            typeinfer.copy_type(self.src, self.dst, loc=self.loc)
            typeinfer.refine_map[self.dst] = self

    def refine(self, typeinfer, target_type):
        assert target_type.is_precise()
        typeinfer.add_type(self.src, target_type, unless_locked=True, loc=self.loc)