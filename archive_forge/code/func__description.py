from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _description(self, optName):
    if self._descr is not None:
        return f'{self._descr} ({self._globPattern})'
    else:
        return f'{optName} ({self._globPattern})'