import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class SingleAlternativeGrandchildOverride(SingleAlternativeChild):

    def my_method(self, arg, kw=99):
        """my_method override."""
        return 'override2'

    def alt(self):
        """Unneeded alternative method."""