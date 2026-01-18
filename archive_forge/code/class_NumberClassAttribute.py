import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_getattr
class NumberClassAttribute(AttributeTemplate):
    key = types.NumberClass

    def resolve___call__(self, classty):
        """
        Resolve a NumPy number class's constructor (e.g. calling numpy.int32(...))
        """
        ty = classty.instance_type

        def typer(val):
            if isinstance(val, (types.BaseTuple, types.Sequence)):
                fnty = self.context.resolve_value_type(np.array)
                sig = fnty.get_call_type(self.context, (val, types.DType(ty)), {})
                return sig.return_type
            elif isinstance(val, (types.Number, types.Boolean, types.IntEnumMember)):
                return ty
            elif isinstance(val, (types.NPDatetime, types.NPTimedelta)):
                if ty.bitwidth == 64:
                    return ty
                else:
                    msg = f'Cannot cast {val} to {ty} as {ty} is not 64 bits wide.'
                    raise errors.TypingError(msg)
            elif isinstance(val, types.Array) and val.ndim == 0 and (val.dtype == ty):
                return ty
            else:
                msg = f'Casting {val} to {ty} directly is unsupported.'
                if isinstance(val, types.Array):
                    msg += f" Try doing '<array>.astype(np.{ty})' instead"
                raise errors.TypingError(msg)
        return types.Function(make_callable_template(key=ty, typer=typer))