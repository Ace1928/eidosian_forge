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
class BuildMapConstraint(object):

    def __init__(self, target, items, special_value, value_indexes, loc):
        self.target = target
        self.items = items
        self.special_value = special_value
        self.value_indexes = value_indexes
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of dict at {0}', self.loc):
            typevars = typeinfer.typevars
            tsets = [(typevars[k.name].getone(), typevars[v.name].getone()) for k, v in self.items]
            if not tsets:
                typeinfer.add_type(self.target, types.DictType(types.undefined, types.undefined, self.special_value), loc=self.loc)
            else:
                ktys = [x[0] for x in tsets]
                vtys = [x[1] for x in tsets]
                strkey = all([isinstance(x, types.StringLiteral) for x in ktys])
                literalvty = all([isinstance(x, types.Literal) for x in vtys])
                vt0 = types.unliteral(vtys[0])

                def check(other):
                    conv = typeinfer.context.can_convert(other, vt0)
                    return conv is not None and conv < Conversion.unsafe
                homogeneous = all([check(types.unliteral(x)) for x in vtys])
                if len(vtys) == 1:
                    valty = vtys[0]
                    if isinstance(valty, (types.LiteralStrKeyDict, types.List, types.LiteralList)):
                        homogeneous = False
                if strkey and (not homogeneous):
                    resolved_dict = {x: y for x, y in zip(ktys, vtys)}
                    ty = types.LiteralStrKeyDict(resolved_dict, self.value_indexes)
                    typeinfer.add_type(self.target, ty, loc=self.loc)
                else:
                    init_value = self.special_value if literalvty else None
                    key_type, value_type = tsets[0]
                    typeinfer.add_type(self.target, types.DictType(key_type, value_type, init_value), loc=self.loc)