import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def _common_prefix(self, m):
    """Given a list of str, returns the longest common prefix.

    Args:
      m: (list of str) A list of strings.

    Returns:
      (str) The longest common prefix.
    """
    if not m:
        return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1