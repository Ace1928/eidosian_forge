import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method3_with_2(self):
    return '3-2 ' + self.method2()