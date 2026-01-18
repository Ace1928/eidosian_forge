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
class MySQLConnectorDatabase(MySQLDatabase):

    def _connect(self):
        if mysql_connector is None:
            raise ImproperlyConfigured('MySQL connector not installed!')
        return mysql_connector.connect(db=self.database, autocommit=True, **self.connect_params)

    def cursor(self, commit=None, named_cursor=None):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        if self.is_closed():
            if self.autoconnect:
                self.connect()
            else:
                raise InterfaceError('Error, database connection not opened.')
        return self._state.conn.cursor(buffered=True)

    def get_binary_type(self):
        return mysql_connector.Binary