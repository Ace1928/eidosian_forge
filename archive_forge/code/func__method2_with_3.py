import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _method2_with_3(self):
    return '2-3 ' + self.method3()