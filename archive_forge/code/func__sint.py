import os
import pytest
import textwrap
import numpy as np
from . import util
@staticmethod
def _sint(s, start=0, end=None):
    """Return the content of a string buffer as integer value.

        For example:
          _sint('1234') -> 4321
          _sint('123A') -> 17321
        """
    if isinstance(s, np.ndarray):
        s = s.tobytes()
    elif isinstance(s, str):
        s = s.encode()
    assert isinstance(s, bytes)
    if end is None:
        end = len(s)
    i = 0
    for j in range(start, min(end, len(s))):
        i += s[j] * 10 ** j
    return i