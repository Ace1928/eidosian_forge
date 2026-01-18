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
def _retrieve_proxy_for_target(cls, target: _PT, regenerate: bool=True) -> Optional[Self]:
    try:
        proxy_ref = cls._proxy_objects[weakref.ref(target)]
    except KeyError:
        pass
    else:
        proxy = proxy_ref()
        if proxy is not None:
            return proxy
    if regenerate:
        return cls._regenerate_proxy_for_target(target)
    else:
        return None