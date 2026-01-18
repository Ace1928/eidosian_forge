import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPool_5(self, res):
    self.flushLoggedErrors()
    sql = 'select * from NOTABLE'
    d = defer.maybeDeferred(self.dbpool.runQuery, sql)
    d.addCallbacks(lambda res: self.fail('no exception'), lambda f: self.assertFalse(f.check(ConnectionLost)))
    return d