from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
class _ShouldSkipException(Exception):
    """
    Raised when a subtitle should be skipped.
    """