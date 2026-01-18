from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def _getSummary(self):
    """
        Return a formatted count of tests status results.
        """
    summaries = []
    for stat in ('skips', 'expectedFailures', 'failures', 'errors', 'unexpectedSuccesses'):
        num = len(getattr(self, stat))
        if num:
            summaries.append('%s=%d' % (stat, num))
    if self.successes:
        summaries.append('successes=%d' % (self.successes,))
    summary = summaries and ' (' + ', '.join(summaries) + ')' or ''
    return summary