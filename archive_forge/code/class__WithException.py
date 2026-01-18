import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
class _WithException(Protocol[_F, _ET]):
    Exception: _ET
    __call__: _F