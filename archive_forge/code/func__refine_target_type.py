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
def _refine_target_type(self, typeinfer, targetty, idxty, valty, sig):
    """Refine the target-type given the known index type and value type.
        """
    if _is_array_not_precise(targetty):
        typeinfer.add_type(self.target.name, sig.args[0], loc=self.loc)
    if isinstance(targetty, types.DictType):
        if not targetty.is_precise():
            refined = targetty.refine(idxty, valty)
            typeinfer.add_type(self.target.name, refined, loc=self.loc)
        elif isinstance(targetty, types.LiteralStrKeyDict):
            typeinfer.add_type(self.target.name, types.DictType(idxty, valty), loc=self.loc)