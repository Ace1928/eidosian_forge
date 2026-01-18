from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
import math
from numbers import Complex
import pprint
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import final
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import _pytest._code
from _pytest.outcomes import fail
def _approx_scalar(self, x) -> 'ApproxScalar':
    if isinstance(x, Decimal):
        return ApproxDecimal(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)
    return ApproxScalar(x, rel=self.rel, abs=self.abs, nan_ok=self.nan_ok)