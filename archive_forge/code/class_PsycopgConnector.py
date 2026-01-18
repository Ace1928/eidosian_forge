import os
import stat
from typing import Dict, Optional
from twisted.enterprise.adbapi import (
from twisted.internet import defer, interfaces, reactor
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.trial import unittest
class PsycopgConnector(DBTestConnector):
    TEST_PREFIX = 'Psycopg'

    def can_connect(self):
        try:
            import psycopg
        except BaseException:
            return False
        try:
            conn = psycopg.connect(database=self.DB_NAME, user=self.DB_USER, password=self.DB_PASS)
            conn.close()
            return True
        except BaseException:
            return False

    def getPoolArgs(self):
        args = ('psycopg',)
        kw = {'database': self.DB_NAME, 'user': self.DB_USER, 'password': self.DB_PASS, 'cp_min': 0}
        return (args, kw)