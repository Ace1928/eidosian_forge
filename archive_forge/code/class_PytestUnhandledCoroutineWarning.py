import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
@final
class PytestUnhandledCoroutineWarning(PytestReturnNotNoneWarning):
    """Warning emitted for an unhandled coroutine.

    A coroutine was encountered when collecting test functions, but was not
    handled by any async-aware plugin.
    Coroutine test functions are not natively supported.
    """
    __module__ = 'pytest'