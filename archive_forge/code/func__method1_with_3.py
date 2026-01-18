import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method1_with_3(self):
    return '1-3 ' + self.method3()