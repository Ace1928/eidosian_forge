import enum
import inspect
import warnings
from functools import wraps
from typing import Callable, Optional
from .logging import get_logger
class DeprecatedEnum(enum.Enum, metaclass=OnAccess):
    """
    Enum class that calls `deprecate` method whenever a member is accessed.
    """

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member._on_access = member.deprecate
        return member

    @property
    def help_message(self):
        return ''

    def deprecate(self):
        help_message = f' {self.help_message}' if self.help_message else ''
        warnings.warn(f"'{self.__objclass__.__name__}' is deprecated and will be removed in the next major version of datasets." + help_message, FutureWarning, stacklevel=3)