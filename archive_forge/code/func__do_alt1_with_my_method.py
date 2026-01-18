import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _do_alt1_with_my_method(self):
    return 'reverse ' + self.my_method(0, kw=0)