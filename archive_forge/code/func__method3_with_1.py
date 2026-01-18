import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method3_with_1(self):
    return '3-1 ' + self.method1()