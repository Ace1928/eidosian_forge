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
class PooledPsycopg3Database(PooledDatabase, Psycopg3Database):

    def _is_closed(self, conn):
        if conn.closed:
            return True
        txn_status = conn.pgconn.transaction_status
        if txn_status == conn.TransactionStatus.UNKNOWN:
            return True
        elif txn_status != conn.TransactionStatus.IDLE:
            conn.rollback()
        return False

    def _can_reuse(self, conn):
        txn_status = conn.pgconn.transaction_status
        if txn_status == conn.TransactionStatus.UNKNOWN:
            return False
        elif txn_status == conn.TransactionStatus.INERROR:
            conn.reset()
        elif txn_status != conn.TransactionStatus.IDLE:
            conn.rollback()
        return True