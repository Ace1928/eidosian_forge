import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class MySQLConnector(DBTestConnector):
    TEST_PREFIX = 'MySQL'
    trailing_spaces_ok = False
    can_rollback = False
    early_reconnect = False

    def can_connect(self):
        try:
            import MySQLdb
        except BaseException:
            return False
        try:
            conn = MySQLdb.connect(db=self.DB_NAME, user=self.DB_USER, passwd=self.DB_PASS)
            conn.close()
            return True
        except BaseException:
            return False

    def getPoolArgs(self):
        args = ('MySQLdb',)
        kw = {'db': self.DB_NAME, 'user': self.DB_USER, 'passwd': self.DB_PASS}
        return (args, kw)