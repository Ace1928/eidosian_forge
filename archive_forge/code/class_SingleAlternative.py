import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class SingleAlternative(metaclass=ABCMetaImplementAnyOneOf):

    def _default_impl(self, arg, kw=99):
        """Default implementation."""

    @alternative(requires='alt', implementation=_default_impl)
    def my_method(self, arg, kw=99) -> None:
        """my_method doc."""

    @abc.abstractmethod
    def alt(self) -> None:
        pass