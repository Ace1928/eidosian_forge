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
class MariaDBConnectorDatabase(MySQLDatabase):

    def _connect(self):
        if mariadb is None:
            raise ImproperlyConfigured('mariadb connector not installed!')
        self.connect_params.pop('charset', None)
        self.connect_params.pop('sql_mode', None)
        self.connect_params.pop('use_unicode', None)
        return mariadb.connect(db=self.database, autocommit=True, **self.connect_params)

    def cursor(self, commit=None, named_cursor=None):
        if commit is not None:
            __deprecated__('"commit" has been deprecated and is a no-op.')
        if self.is_closed():
            if self.autoconnect:
                self.connect()
            else:
                raise InterfaceError('Error, database connection not opened.')
        return self._state.conn.cursor(buffered=True)

    def _set_server_version(self, conn):
        version = conn.server_version
        version, point = divmod(version, 100)
        version, minor = divmod(version, 100)
        self.server_version = (version, minor, point)
        if self.server_version >= (10, 5, 0):
            self.returning_clause = True

    def last_insert_id(self, cursor, query_type=None):
        if not self.returning_clause:
            return cursor.lastrowid
        elif query_type == Insert.SIMPLE:
            try:
                return cursor[0][0]
            except (AttributeError, IndexError):
                return cursor.lastrowid
        return cursor

    def get_binary_type(self):
        return mariadb.Binary