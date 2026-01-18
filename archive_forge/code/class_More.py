import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
class More(PagerCommand):
    """The pager command ``more``."""

    def command(self) -> List[str]:
        if sys.platform.startswith('win32'):
            return ['more.com']
        return ['more']

    def environment_variables(self, config: PagerConfig) -> Optional[Dict[str, str]]:
        return None