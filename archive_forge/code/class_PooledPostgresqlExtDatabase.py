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
class PooledPostgresqlExtDatabase(_PooledPostgresqlDatabase, PostgresqlExtDatabase):
    pass