import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class SingleAlternativeOverride(SingleAlternative):

    def my_method(self, arg, kw=99) -> None:
        """my_method override."""

    def alt(self) -> None:
        """Unneeded alternative method."""