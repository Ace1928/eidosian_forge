from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
class LazyIbisAttr(LazyIbisObject):

    def __init__(self, parent: LazyIbisObject, name: str):
        super().__init__()
        self._super_lazy_internal_ctx.update(parent._super_lazy_internal_ctx)
        self._super_lazy_internal_objs: Dict[str, Any] = dict(parent=parent, name=name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return LazyIbisFunction(self._super_lazy_internal_objs['parent'], self._super_lazy_internal_objs['name'], *args, **kwargs)