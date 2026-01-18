import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class NonThreadPool:

    def callInThreadWithCallback(self, onResult, f, *a, **kw):
        success = True
        try:
            result = f(*a, **kw)
        except Exception:
            success = False
            result = Failure()
        onResult(success, result)