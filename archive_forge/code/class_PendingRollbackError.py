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
class PendingRollbackError(InvalidRequestError):
    """A transaction has failed and needs to be rolled back before
    continuing.

    .. versionadded:: 1.4

    """