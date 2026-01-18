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
class PytestCollectionWarning(PytestWarning):
    """Warning emitted when pytest is not able to collect a file or symbol in a module."""
    __module__ = 'pytest'