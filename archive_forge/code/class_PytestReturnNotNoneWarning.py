import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
class PytestReturnNotNoneWarning(PytestWarning):
    """Warning emitted when a test function is returning value other than None."""
    __module__ = 'pytest'