import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method2_with_1(self):
    return '2-1 ' + self.method1()