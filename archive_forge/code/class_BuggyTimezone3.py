from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
class BuggyTimezone3(datetime.tzinfo):
    """
        Wrong timezone name type
        """

    def tzname(self, dt):
        return 240

    def utcoffset(self, dt):
        return None