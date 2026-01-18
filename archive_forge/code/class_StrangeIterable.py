import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
class StrangeIterable:

    def __init__(self, lst):
        self.lst = lst

    def __iter__(self):
        return self.lst.__iter__()