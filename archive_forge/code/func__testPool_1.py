import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def _testPool_1(self, res):
    sql = 'select count(1) from simple'
    d = self.dbpool.runQuery(sql)

    def _check(row):
        self.assertTrue(int(row[0][0]) == 0, 'Table not empty')
    d.addCallback(_check)
    return d