import abc
from typing import NoReturn, Optional
import pytest
from cirq import ABCMetaImplementAnyOneOf, alternative
class SingleAlternativeGrandchild(SingleAlternativeChild):

    def alt(self):
        return 'alt2'