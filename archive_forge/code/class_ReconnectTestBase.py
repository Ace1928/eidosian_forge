import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class ReconnectTestBase:
    """
    Test the asynchronous DB-API code with reconnect.
    """
    if interfaces.IReactorThreads(reactor, None) is None:
        skip = 'ADB-API requires threads, no way to test without them'

    def extraSetUp(self):
        """
        Skip the test if C{good_sql} is unavailable.  Otherwise, set up the
        database, create a connection pool pointed at it, and set up a simple
        schema in it.
        """
        if self.good_sql is None:
            raise unittest.SkipTest('no good sql for reconnect test')
        self.startDB()
        self.dbpool = self.makePool(cp_max=1, cp_reconnect=True, cp_good_sql=self.good_sql)
        self.dbpool.start()
        return self.dbpool.runOperation(simple_table_schema)

    def tearDown(self):
        d = self.dbpool.runOperation('DROP TABLE simple')
        d.addCallback(lambda res: self.dbpool.close())
        d.addCallback(lambda res: self.stopDB())
        return d

    def test_pool(self):
        d = defer.succeed(None)
        d.addCallback(self._testPool_1)
        d.addCallback(self._testPool_2)
        if not self.early_reconnect:
            d.addCallback(self._testPool_3)
        d.addCallback(self._testPool_4)
        d.addCallback(self._testPool_5)
        return d

    def _testPool_1(self, res):
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            self.assertTrue(int(row[0][0]) == 0, 'Table not empty')
        d.addCallback(_check)
        return d

    def _testPool_2(self, res):
        list(self.dbpool.connections.values())[0].close()

    def _testPool_3(self, res):
        sql = 'select count(1) from simple'
        d = defer.maybeDeferred(self.dbpool.runQuery, sql)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: None)
        return d

    def _testPool_4(self, res):
        sql = 'select count(1) from simple'
        d = self.dbpool.runQuery(sql)

        def _check(row):
            self.assertTrue(int(row[0][0]) == 0, 'Table not empty')
        d.addCallback(_check)
        return d

    def _testPool_5(self, res):
        self.flushLoggedErrors()
        sql = 'select * from NOTABLE'
        d = defer.maybeDeferred(self.dbpool.runQuery, sql)
        d.addCallbacks(lambda res: self.fail('no exception'), lambda f: self.assertFalse(f.check(ConnectionLost)))
        return d