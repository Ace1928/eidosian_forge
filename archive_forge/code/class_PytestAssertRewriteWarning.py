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
class PytestAssertRewriteWarning(PytestWarning):
    """Warning emitted by the pytest assert rewrite module."""
    __module__ = 'pytest'