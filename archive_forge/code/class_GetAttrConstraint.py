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
class GetAttrConstraint(object):

    def __init__(self, target, attr, value, loc, inst):
        self.target = target
        self.attr = attr
        self.value = value
        self.loc = loc
        self.inst = inst

    def __call__(self, typeinfer):
        with new_error_context('typing of get attribute at {0}', self.loc):
            typevars = typeinfer.typevars
            valtys = typevars[self.value.name].get()
            for ty in valtys:
                attrty = typeinfer.context.resolve_getattr(ty, self.attr)
                if attrty is None:
                    raise UntypedAttributeError(ty, self.attr, loc=self.inst.loc)
                else:
                    assert attrty.is_precise()
                    typeinfer.add_type(self.target, attrty, loc=self.loc)
            typeinfer.refine_map[self.target] = self

    def refine(self, typeinfer, target_type):
        if isinstance(target_type, types.BoundFunction):
            recvr = target_type.this
            assert recvr.is_precise()
            typeinfer.add_type(self.value.name, recvr, loc=self.loc)
            source_constraint = typeinfer.refine_map.get(self.value.name)
            if source_constraint is not None:
                source_constraint.refine(typeinfer, recvr)

    def __repr__(self):
        return 'resolving type of attribute "{attr}" of "{value}"'.format(value=self.value, attr=self.attr)