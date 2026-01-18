import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class SQLite3Connector(DBTestConnector):
    """
    Connector that uses the stdlib SQLite3 database support.
    """
    TEST_PREFIX = 'SQLite3'
    escape_slashes = False
    num_iterations = 1

    def can_connect(self):
        if requireModule('sqlite3') is None:
            return False
        else:
            return True

    def startDB(self):
        self.database = os.path.join(self.DB_DIR, self.DB_NAME)
        if os.path.exists(self.database):
            os.unlink(self.database)

    def getPoolArgs(self):
        args = ('sqlite3',)
        kw = {'database': self.database, 'cp_max': 1, 'check_same_thread': False}
        return (args, kw)