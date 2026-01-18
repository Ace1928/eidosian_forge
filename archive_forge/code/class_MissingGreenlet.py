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
class MissingGreenlet(InvalidRequestError):
    """Error raised by the async greenlet await\\_ if called while not inside
    the greenlet spawn context.

    """
    code = 'xd2s'