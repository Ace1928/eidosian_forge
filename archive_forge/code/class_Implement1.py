import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class Implement1(AnyOneAbc):

    def method1(self):
        """Method1 child."""
        return 'child1'