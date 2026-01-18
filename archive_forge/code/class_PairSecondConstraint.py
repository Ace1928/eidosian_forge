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
class PairSecondConstraint(object):

    def __init__(self, target, pair, loc):
        self.target = target
        self.pair = pair
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of pair-second at {0}', self.loc):
            typevars = typeinfer.typevars
            for tp in typevars[self.pair.name].get():
                if not isinstance(tp, types.Pair):
                    continue
                assert tp.second_type.is_precise()
                typeinfer.add_type(self.target, tp.second_type, loc=self.loc)