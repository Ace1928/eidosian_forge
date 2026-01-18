import logging
import warnings
from functools import wraps
from platform import python_version
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec, overload
class WarningCache(set):
    """Cache for warnings."""

    def warn(self, message: str, stacklevel: int=5, **kwargs: Any) -> None:
        """Trigger warning message."""
        if message not in self:
            self.add(message)
            rank_zero_warn(message, stacklevel=stacklevel, **kwargs)

    def deprecation(self, message: str, stacklevel: int=6, **kwargs: Any) -> None:
        """Trigger deprecation message."""
        if message not in self:
            self.add(message)
            rank_zero_deprecation(message, stacklevel=stacklevel, **kwargs)

    def info(self, message: str, stacklevel: int=5, **kwargs: Any) -> None:
        """Trigger info message."""
        if message not in self:
            self.add(message)
            rank_zero_info(message, stacklevel=stacklevel, **kwargs)