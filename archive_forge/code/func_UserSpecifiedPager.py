import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
def UserSpecifiedPager(*env_vars: str) -> PagerCommand:
    """
    Return the pager command for the current environment.

    Each of the specified environment variables is searched in order; the first
    one that is set will be used as the pager command. If none of the
    environment variables is set, the default pager for the platform will be
    used.
    """
    for env_var in env_vars:
        env_pager = os.getenv(env_var)
        if env_pager:
            return CustomPager(env_pager)
    return PlatformPager()