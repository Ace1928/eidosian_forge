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
class ResourceClosedError(InvalidRequestError):
    """An operation was requested from a connection, cursor, or other
    object that's in a closed state."""