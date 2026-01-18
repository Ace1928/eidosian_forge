import io
import logging
import sys
from collections.abc import Sequence
from typing import Optional, List, TextIO
from pyomo.common.config import (
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer
def TextIO_or_Logger(val):
    ans = []
    if not isinstance(val, Sequence):
        val = [val]
    for v in val:
        if v.__class__ in native_logical_types:
            if v:
                ans.append(sys.stdout)
        elif isinstance(v, io.TextIOBase):
            ans.append(v)
        elif isinstance(v, logging.Logger):
            ans.append(LogStream(level=logging.INFO, logger=v))
        else:
            raise ValueError('Expected bool, TextIOBase, or Logger, but received {v.__class__}')
    return ans