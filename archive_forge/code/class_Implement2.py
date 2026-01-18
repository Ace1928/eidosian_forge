import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class Implement2(AnyOneAbc):

    def method2(self):
        """Method2 child."""
        return 'child2'