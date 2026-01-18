import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
def _register_stream(self, stream):
    i = len(self.input_streams)

    def perform_update(stream_index=i, **kwargs):
        if stream_index in self._updating:
            return
        if self.exclusive:
            for j, input_stream in enumerate(self.input_streams):
                if stream_index != j:
                    input_stream.reset()
                    self._updating.add(j)
                    try:
                        input_stream.event()
                    finally:
                        self._updating.remove(j)
        self.event()
    stream.add_subscriber(perform_update)
    self.input_streams.append(stream)