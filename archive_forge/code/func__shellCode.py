from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _shellCode(self, optName, shellType):
    if shellType == _ZSH:
        return '{}:{}:_net_interfaces'.format(self._repeatFlag, self._description(optName))
    raise NotImplementedError(f'Unknown shellType {shellType!r}')