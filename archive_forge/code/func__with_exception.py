import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
def _with_exception(exception_type: _ET) -> Callable[[_F], _WithException[_F, _ET]]:

    def decorate(func: _F) -> _WithException[_F, _ET]:
        func_with_exception = cast(_WithException[_F, _ET], func)
        func_with_exception.Exception = exception_type
        return func_with_exception
    return decorate