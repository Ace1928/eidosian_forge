import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
class StringAnnotation:

    def heap(self) -> 'HeapType':
        return HeapType()

    def copy(self) -> 'StringAnnotation':
        return StringAnnotation()