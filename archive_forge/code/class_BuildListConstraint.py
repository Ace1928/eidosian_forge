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
class BuildListConstraint(_BuildContainerConstraint):

    def __init__(self, target, items, loc):
        self.target = target
        self.items = items
        self.loc = loc

    def __call__(self, typeinfer):
        with new_error_context('typing of {0} at {1}', types.List, self.loc):
            typevars = typeinfer.typevars
            tsets = [typevars[i.name].get() for i in self.items]
            if not tsets:
                typeinfer.add_type(self.target, types.List(types.undefined), loc=self.loc)
            else:
                for typs in itertools.product(*tsets):
                    unified = typeinfer.context.unify_types(*typs)
                    if unified is not None:
                        islit = [isinstance(x, types.Literal) for x in typs]
                        iv = None
                        if all(islit):
                            iv = [x.literal_value for x in typs]
                        typeinfer.add_type(self.target, types.List(unified, initial_value=iv), loc=self.loc)
                    else:
                        typeinfer.add_type(self.target, types.LiteralList(typs), loc=self.loc)