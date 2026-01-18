import datetime
import decimal
import sys
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase
class SqlCipherExtDatabase(_SqlCipherDatabase, SqliteExtDatabase):
    pass