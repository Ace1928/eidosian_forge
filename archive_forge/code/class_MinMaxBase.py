import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
class MinMaxBase(AbstractTemplate):

    def _unify_minmax(self, tys):
        for ty in tys:
            if not isinstance(ty, (types.Number, types.NPDatetime, types.NPTimedelta)):
                return
        return self.context.unify_types(*tys)

    def generic(self, args, kws):
        """
        Resolve a min() or max() call.
        """
        assert not kws
        if not args:
            return
        if len(args) == 1:
            if isinstance(args[0], types.BaseTuple):
                tys = list(args[0])
                if not tys:
                    raise TypeError('%s() argument is an empty tuple' % (self.key.__name__,))
            else:
                return
        else:
            tys = args
        retty = self._unify_minmax(tys)
        if retty is not None:
            return signature(retty, *args)