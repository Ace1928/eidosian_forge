import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
def _stopLogging(self):
    """
        Clean-up hook for fixing potentially global state.  Only for testing of
        this module itself.  If you want less global state, use the new
        warnings system in L{twisted.logger}.
        """
    if self._warningsModule.showwarning == self.showwarning:
        self._warningsModule.showwarning = self._oldshowwarning