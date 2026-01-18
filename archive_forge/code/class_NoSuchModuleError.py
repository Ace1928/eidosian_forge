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
class NoSuchModuleError(ArgumentError):
    """Raised when a dynamically-loaded module (usually a database dialect)
    of a particular name cannot be located."""