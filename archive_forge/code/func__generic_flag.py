from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def _generic_flag(self, flagName, value=None):
    if value not in ('', None):
        raise UsageError('Flag \'%s\' takes no argument. Not even "%s".' % (flagName, value))
    self.opts[flagName] = 1