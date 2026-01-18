import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def _add_builtin_meta_specs():
    for name, spec in globals().items():
        if name.startswith('MetaSpec_'):
            add_meta_spec(spec)