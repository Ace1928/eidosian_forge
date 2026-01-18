from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
class SafeConfigParser(ConfigParser):
    """ConfigParser alias for backwards compatibility purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('The SafeConfigParser class has been renamed to ConfigParser in Python 3.2. This alias will be removed in Python 3.12. Use ConfigParser directly instead.', DeprecationWarning, stacklevel=2)