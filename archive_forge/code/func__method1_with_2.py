import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method1_with_2(self):
    return '1-2 ' + self.method2()