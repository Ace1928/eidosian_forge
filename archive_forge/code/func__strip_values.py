import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _strip_values(values):
    """Strip reflected values quotes"""
    strip_values = []
    for a in values:
        if a[0:1] == '"' or a[0:1] == "'":
            a = a[1:-1].replace(a[0] * 2, a[0])
        strip_values.append(a)
    return strip_values