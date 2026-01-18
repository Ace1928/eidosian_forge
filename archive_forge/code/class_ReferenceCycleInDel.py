import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
class ReferenceCycleInDel:
    """
            An object that not only contains a reference cycle, but creates new
            cycles whenever it's garbage-collected and its __del__ runs
            """
    make_cycle = True

    def __init__(self):
        self.cycle = self

    def __del__(self):
        self.cycle = None
        if ReferenceCycleInDel.make_cycle:
            ReferenceCycleInDel()