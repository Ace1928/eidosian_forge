import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPool_1_4(self, res):
    d = defer.maybeDeferred(self.dbpool.runWithConnection, self.bad_withConnection)
    d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
    return d