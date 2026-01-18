from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CompleteMultiList(Completer):
    """
    Completes multiple comma-separated items based on a fixed list of words
    """

    def __init__(self, items, **kw):
        Completer.__init__(self, **kw)
        self._items = items

    def _shellCode(self, optName, shellType):
        if shellType == _ZSH:
            return "{}:{}:_values -s , '{}' {}".format(self._repeatFlag, self._description(optName), self._description(optName), ' '.join(self._items))
        raise NotImplementedError(f'Unknown shellType {shellType!r}')