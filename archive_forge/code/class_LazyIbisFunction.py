from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
class LazyIbisFunction(LazyIbisObject):

    def __init__(self, obj: LazyIbisObject, func: str, *args: Any, **kwargs: Any):
        super().__init__()
        self._super_lazy_internal_ctx.update(obj._super_lazy_internal_ctx)
        for x in args:
            if isinstance(x, LazyIbisObject):
                self._super_lazy_internal_ctx.update(x._super_lazy_internal_ctx)
        for x in kwargs.values():
            if isinstance(x, LazyIbisObject):
                self._super_lazy_internal_ctx.update(x._super_lazy_internal_ctx)
        self._super_lazy_internal_objs: Dict[str, Any] = dict(obj=obj, func=func, args=args, kwargs=kwargs)