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
class PytestUnhandledThreadExceptionWarning(PytestWarning):
    """An unhandled exception occurred in a :class:`~threading.Thread`.

    Such exceptions don't propagate normally.
    """
    __module__ = 'pytest'