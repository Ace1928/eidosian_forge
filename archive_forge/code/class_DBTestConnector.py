import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class DBTestConnector:
    """
    A class which knows how to test for the presence of
    and establish a connection to a relational database.

    To enable test cases  which use a central, system database,
    you must create a database named DB_NAME with a user DB_USER
    and password DB_PASS with full access rights to database DB_NAME.
    """
    TEST_PREFIX: Optional[str] = None
    DB_NAME = 'twisted_test'
    DB_USER = 'twisted_test'
    DB_PASS = 'twisted_test'
    DB_DIR = None
    nulls_ok = True
    trailing_spaces_ok = True
    can_rollback = True
    test_failures = True
    escape_slashes = True
    good_sql: Optional[str] = ConnectionPool.good_sql
    early_reconnect = True
    can_clear = True
    num_iterations = 50

    def setUp(self):
        self.DB_DIR = self.mktemp()
        os.mkdir(self.DB_DIR)
        if not self.can_connect():
            raise unittest.SkipTest('%s: Cannot access db' % self.TEST_PREFIX)
        return self.extraSetUp()

    def can_connect(self):
        """Return true if this database is present on the system
        and can be used in a test."""
        raise NotImplementedError()

    def startDB(self):
        """Take any steps needed to bring database up."""
        pass

    def stopDB(self):
        """Bring database down, if needed."""
        pass

    def makePool(self, **newkw):
        """Create a connection pool with additional keyword arguments."""
        args, kw = self.getPoolArgs()
        kw = kw.copy()
        kw.update(newkw)
        return ConnectionPool(*args, **kw)

    def getPoolArgs(self):
        """Return a tuple (args, kw) of list and keyword arguments
        that need to be passed to ConnectionPool to create a connection
        to this database."""
        raise NotImplementedError()