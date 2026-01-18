from __future__ import annotations
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
from .util import compat
from .util import preloaded as _preloaded
class NoForeignKeysError(ArgumentError):
    """Raised when no foreign keys can be located between two selectables
    during a join."""