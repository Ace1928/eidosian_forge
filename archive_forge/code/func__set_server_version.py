import json
from peewee import ImproperlyConfigured
from peewee import Insert
from peewee import MySQLDatabase
from peewee import Node
from peewee import NodeList
from peewee import SQL
from peewee import TextField
from peewee import fn
from peewee import __deprecated__
def _set_server_version(self, conn):
    version = conn.server_version
    version, point = divmod(version, 100)
    version, minor = divmod(version, 100)
    self.server_version = (version, minor, point)
    if self.server_version >= (10, 5, 0):
        self.returning_clause = True