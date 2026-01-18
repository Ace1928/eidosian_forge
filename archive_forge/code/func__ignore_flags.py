import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _ignore_flags(self, *flags):
    """Ignore the flags, if they are raised"""
    self._ignored_flags = self._ignored_flags + list(flags)
    return list(flags)