import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def _uniformize(val):
    if isinstance(val, dict):
        return {k: _uniformize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return tuple((_uniformize(el) for el in val))
    return val