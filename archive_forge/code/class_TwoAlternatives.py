import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class TwoAlternatives(metaclass=ABCMetaImplementAnyOneOf):

    def _default_impl1(self, arg, kw=99):
        return f'default1({arg}, {kw}) ' + self.alt1()

    def _default_impl2(self, arg, kw=99):
        return f'default2({arg}, {kw}) ' + self.alt2()

    @alternative(requires='alt1', implementation=_default_impl1)
    @alternative(requires='alt2', implementation=_default_impl2)
    def my_method(self, arg, kw=99) -> str:
        """Docstring."""
        raise NotImplementedError

    @abc.abstractmethod
    def alt1(self) -> Optional[str]:
        pass

    @abc.abstractmethod
    def alt2(self) -> Optional[str]:
        pass