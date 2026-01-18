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
def _add_refine_map(self, typeinfer, typevars, sig):
    """Add this expression to the refine_map base on the type of target_type
        """
    target_type = typevars[self.target].getone()
    if isinstance(target_type, types.Array) and isinstance(sig.return_type.dtype, types.Undefined):
        typeinfer.refine_map[self.target] = self
    if isinstance(target_type, types.DictType) and (not target_type.is_precise()):
        typeinfer.refine_map[self.target] = self