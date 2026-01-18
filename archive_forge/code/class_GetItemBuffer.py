import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer_global(operator.getitem)
class GetItemBuffer(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        [ary, idx] = args
        out = get_array_index_type(ary, idx)
        if out is not None:
            return signature(out.result, ary, out.index)