from __future__ import annotations
import re
from typing import Any
from typing import Optional
from typing import TypeVar
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import OVERLAP
from ... import types as sqltypes
from ... import util
from ...sql import expression
from ...sql import operators
from ...sql._typing import _TypeEngineArgument
@util.memoized_property
def _against_native_enum(self):
    return isinstance(self.item_type, sqltypes.Enum) and self.item_type.native_enum