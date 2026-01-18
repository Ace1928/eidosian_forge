import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class Implement3(AnyOneAbc):

    def method3(self):
        """Method3 child."""
        return 'child3'