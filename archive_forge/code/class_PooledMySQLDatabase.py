import functools
import heapq
import logging
import random
import threading
import time
from collections import namedtuple
from itertools import chain
from peewee import MySQLDatabase
from peewee import PostgresqlDatabase
from peewee import SqliteDatabase
class PooledMySQLDatabase(PooledDatabase, MySQLDatabase):

    def _is_closed(self, conn):
        if self.server_version[0] == 8:
            args = ()
        else:
            args = (False,)
        try:
            conn.ping(*args)
        except:
            return True
        else:
            return False