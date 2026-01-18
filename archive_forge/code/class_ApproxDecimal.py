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
class ApproxDecimal(ApproxScalar):
    """Perform approximate comparisons where the expected value is a Decimal."""
    DEFAULT_ABSOLUTE_TOLERANCE = Decimal('1e-12')
    DEFAULT_RELATIVE_TOLERANCE = Decimal('1e-6')