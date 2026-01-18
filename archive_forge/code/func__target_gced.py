from __future__ import annotations
import abc
import functools
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generator
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
import weakref
from . import exc as async_exc
from ... import util
from ...util.typing import Literal
from ...util.typing import Self
@classmethod
def _target_gced(cls, ref: weakref.ref[_PT], proxy_ref: Optional[weakref.ref[Self]]=None) -> None:
    cls._proxy_objects.pop(ref, None)