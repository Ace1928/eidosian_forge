from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
class CompleteHostnames(Completer):
    """
    Complete hostnames
    """

    def _shellCode(self, optName, shellType):
        if shellType == _ZSH:
            return f'{self._repeatFlag}:{self._description(optName)}:_hosts'
        raise NotImplementedError(f'Unknown shellType {shellType!r}')