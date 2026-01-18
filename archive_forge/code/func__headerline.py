import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
@classmethod
def _headerline(cls, start, name, ts):
    l = start + b' ' + name
    if ts is not None:
        l += b'\t%s' % ts
    l += b'\n'
    return l