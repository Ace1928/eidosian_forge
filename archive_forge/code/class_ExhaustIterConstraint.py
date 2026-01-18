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
class ExhaustIterConstraint(object):

    def __init__(self, target, count, iterator, loc):
        self.target = target
        self.count = count
        self.iterator = iterator
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of exhaust iter at {0}', self.loc):
            typevars = typeinfer.typevars
            for tp in typevars[self.iterator.name].get():
                tp = tp.type if isinstance(tp, types.Optional) else tp
                if isinstance(tp, types.BaseTuple):
                    if len(tp) == self.count:
                        assert tp.is_precise()
                        typeinfer.add_type(self.target, tp, loc=self.loc)
                        break
                    else:
                        msg = (f'wrong tuple length for {self.iterator.name}: ', f'expected {self.count}, got {len(tp)}')
                        raise NumbaValueError(msg)
                elif isinstance(tp, types.IterableType):
                    tup = types.UniTuple(dtype=tp.iterator_type.yield_type, count=self.count)
                    assert tup.is_precise()
                    typeinfer.add_type(self.target, tup, loc=self.loc)
                    break
                else:
                    raise TypingError('failed to unpack {}'.format(tp), loc=self.loc)