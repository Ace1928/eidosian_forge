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
def _raise_for_not_started(self) -> NoReturn:
    raise async_exc.AsyncContextNotStarted('%s context has not been started and object has not been awaited.' % self.__class__.__name__)