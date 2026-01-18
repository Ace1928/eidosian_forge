import numpy
import pytest
from thinc.api import Optimizer, registry
def _test_schedule_valid():
    while True:
        yield 0.456