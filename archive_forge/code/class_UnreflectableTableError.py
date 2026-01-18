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
class UnreflectableTableError(InvalidRequestError):
    """Table exists but can't be reflected for some reason.

    .. versionadded:: 1.2

    """