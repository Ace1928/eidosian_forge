import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def _overload_plus(operator, sleep):
    m1 = create_model(name='a')
    m2 = create_model(name='b')
    with Model.define_operators({operator: lambda a, b: a.name + b.name}):
        time.sleep(sleep)
        if operator == '+':
            value = m1 + m2
        else:
            value = m1 * m2
    assert value == 'ab'
    assert Model._context_operators.get() == {}