import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
def DefaultPager() -> PagerCommand:
    """
    Return the default pager command for the current environment.

    If there is a $PAGER environment variable configured, this command will be
    used. Otherwise, the default pager for the platform will be used.
    """
    if os.name == 'posix':
        if os.getuid() not in {0, os.geteuid()}:
            return PlatformPager()
    return UserSpecifiedPager('PAGER')