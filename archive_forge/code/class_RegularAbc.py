import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class RegularAbc(metaclass=ABCMetaImplementAnyOneOf):

    @abc.abstractmethod
    def my_method(self) -> str:
        """Docstring."""