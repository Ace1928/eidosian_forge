import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPool_7(self, res):
    ds = []
    for i in range(self.num_iterations):
        sql = 'select x from simple where x = %d' % i
        ds.append(self.dbpool.runQuery(sql))
    dlist = defer.DeferredList(ds, fireOnOneErrback=True)

    def _check(result):
        for i in range(self.num_iterations):
            self.assertTrue(result[i][1][0][0] == i, 'Value not returned')
    dlist.addCallback(_check)
    return dlist