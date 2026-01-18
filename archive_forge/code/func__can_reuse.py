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
def _can_reuse(self, conn):
    txn_status = conn.pgconn.transaction_status
    if txn_status == conn.TransactionStatus.UNKNOWN:
        return False
    elif txn_status == conn.TransactionStatus.INERROR:
        conn.reset()
    elif txn_status != conn.TransactionStatus.IDLE:
        conn.rollback()
    return True