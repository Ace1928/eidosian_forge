from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
@extensible_class
class LazyIbisObject:

    def __init__(self, obj: Any=None):
        self._super_lazy_internal_ctx: Dict[int, Any] = {}
        if obj is not None:
            self._super_lazy_internal_ctx[id(self)] = obj

    def __getattr__(self, name: str) -> Any:
        if not name.startswith('_'):
            return LazyIbisAttr(self, name)