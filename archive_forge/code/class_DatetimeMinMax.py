from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
@infer_global(npdatetime_helpers.datetime_minimum)
@infer_global(npdatetime_helpers.datetime_maximum)
class DatetimeMinMax(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        error_msg = 'DatetimeMinMax requires both arguments to be NPDatetime type or both arguments to be NPTimedelta types'
        assert isinstance(args[0], (types.NPDatetime, types.NPTimedelta)), error_msg
        if isinstance(args[0], types.NPDatetime):
            if not isinstance(args[1], types.NPDatetime):
                raise TypeError(error_msg)
        elif not isinstance(args[1], types.NPTimedelta):
            raise TypeError(error_msg)
        return signature(args[0], *args)