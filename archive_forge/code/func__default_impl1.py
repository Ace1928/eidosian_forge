import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
def _default_impl1(self, arg, kw=99):
    return f'default1({arg}, {kw}) ' + self.alt1()