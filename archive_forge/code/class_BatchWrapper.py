from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
class BatchWrapper:

    def __init__(self, batch):
        self.batch = batch

    def __arrow_c_array__(self, requested_schema=None):
        return self.batch.__arrow_c_array__(requested_schema)