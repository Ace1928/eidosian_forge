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
class AwaitRequired(InvalidRequestError):
    """Error raised by the async greenlet spawn if no async operation
    was awaited when it required one.

    """
    code = 'xd1r'