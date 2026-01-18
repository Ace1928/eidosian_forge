import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(type)
class TypeBuiltin(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        if len(args) == 1:
            arg = types.unliteral(args[0])
            classty = self.context.resolve_getattr(arg, '__class__')
            if classty is not None:
                return signature(classty, *args)