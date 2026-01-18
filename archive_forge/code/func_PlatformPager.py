import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
def PlatformPager() -> PagerCommand:
    """
    Return the default pager command for the current platform.
    """
    if sys.platform.startswith('aix'):
        return More()
    if sys.platform.startswith('win32'):
        return More()
    return Less()