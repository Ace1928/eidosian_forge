import json
from peewee import *
from peewee import Expression
from peewee import Node
from peewee import NodeList
from playhouse.postgres_ext import ArrayField
from playhouse.postgres_ext import DateTimeTZField
from playhouse.postgres_ext import IndexedFieldMixin
from playhouse.postgres_ext import IntervalField
from playhouse.postgres_ext import Match
from playhouse.postgres_ext import TSVectorField
from playhouse.postgres_ext import _JsonLookupBase
class Psycopg3Database(PostgresqlDatabase):

    def _connect(self):
        if psycopg is None:
            raise ImproperlyConfigured('psycopg3 is not installed!')
        conn = psycopg.connect(dbname=self.database, **self.connect_params)
        if self._isolation_level is not None:
            conn.isolation_level = self._isolation_level
        conn.autocommit = True
        return conn

    def get_binary_type(self):
        return psycopg.Binary

    def _set_server_version(self, conn):
        self.server_version = conn.pgconn.server_version
        if self.server_version >= 90600:
            self.safe_create_index = True

    def is_connection_usable(self):
        if self._state.closed:
            return False
        conn = self._state.conn
        return conn.pgconn.transaction_status < conn.TransactionStatus.INERROR

    def extract_date(self, date_part, date_field):
        return fn.EXTRACT(NodeList((SQL(date_part), SQL('FROM'), date_field)))