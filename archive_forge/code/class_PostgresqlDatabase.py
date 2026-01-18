from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class PostgresqlDatabase(Database):
    field_types = {'AUTO': 'SERIAL', 'BIGAUTO': 'BIGSERIAL', 'BLOB': 'BYTEA', 'BOOL': 'BOOLEAN', 'DATETIME': 'TIMESTAMP', 'DECIMAL': 'NUMERIC', 'DOUBLE': 'DOUBLE PRECISION', 'UUID': 'UUID', 'UUIDB': 'BYTEA'}
    operations = {'REGEXP': '~', 'IREGEXP': '~*'}
    param = '%s'
    compound_select_parentheses = CSQ_PARENTHESES_ALWAYS
    for_update = True
    nulls_ordering = True
    returning_clause = True
    safe_create_index = False
    sequences = True

    def init(self, database, register_unicode=True, encoding=None, isolation_level=None, **kwargs):
        self._register_unicode = register_unicode
        self._encoding = encoding
        self._isolation_level = isolation_level
        super(PostgresqlDatabase, self).init(database, **kwargs)

    def _connect(self):
        if psycopg2 is None:
            raise ImproperlyConfigured('Postgres driver not installed!')
        params = self.connect_params.copy()
        if self.database.startswith('postgresql://'):
            params.setdefault('dsn', self.database)
        else:
            params.setdefault('dbname', self.database)
        conn = psycopg2.connect(**params)
        if self._register_unicode:
            pg_extensions.register_type(pg_extensions.UNICODE, conn)
            pg_extensions.register_type(pg_extensions.UNICODEARRAY, conn)
        if self._encoding:
            conn.set_client_encoding(self._encoding)
        if self._isolation_level:
            conn.set_isolation_level(self._isolation_level)
        conn.autocommit = True
        return conn

    def _set_server_version(self, conn):
        self.server_version = conn.server_version
        if self.server_version >= 90600:
            self.safe_create_index = True

    def is_connection_usable(self):
        if self._state.closed:
            return False
        txn_status = self._state.conn.get_transaction_status()
        return txn_status < pg_extensions.TRANSACTION_STATUS_INERROR

    def last_insert_id(self, cursor, query_type=None):
        try:
            return cursor if query_type != Insert.SIMPLE else cursor[0][0]
        except (IndexError, KeyError, TypeError):
            pass

    def rows_affected(self, cursor):
        try:
            return cursor.rowcount
        except AttributeError:
            return cursor.cursor.rowcount

    def begin(self, isolation_level=None):
        if self.is_closed():
            self.connect()
        if isolation_level:
            stmt = 'BEGIN TRANSACTION ISOLATION LEVEL %s' % isolation_level
        else:
            stmt = 'BEGIN'
        with __exception_wrapper__:
            self.cursor().execute(stmt)

    def get_tables(self, schema=None):
        query = 'SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s ORDER BY tablename'
        cursor = self.execute_sql(query, (schema or 'public',))
        return [table for table, in cursor.fetchall()]

    def get_views(self, schema=None):
        query = 'SELECT viewname, definition FROM pg_catalog.pg_views WHERE schemaname = %s ORDER BY viewname'
        cursor = self.execute_sql(query, (schema or 'public',))
        return [ViewMetadata(view_name, sql.strip(' \t;')) for view_name, sql in cursor.fetchall()]

    def get_indexes(self, table, schema=None):
        query = "\n            SELECT\n                i.relname, idxs.indexdef, idx.indisunique,\n                array_to_string(ARRAY(\n                    SELECT pg_get_indexdef(idx.indexrelid, k + 1, TRUE)\n                    FROM generate_subscripts(idx.indkey, 1) AS k\n                    ORDER BY k), ',')\n            FROM pg_catalog.pg_class AS t\n            INNER JOIN pg_catalog.pg_index AS idx ON t.oid = idx.indrelid\n            INNER JOIN pg_catalog.pg_class AS i ON idx.indexrelid = i.oid\n            INNER JOIN pg_catalog.pg_indexes AS idxs ON\n                (idxs.tablename = t.relname AND idxs.indexname = i.relname)\n            WHERE t.relname = %s AND t.relkind = %s AND idxs.schemaname = %s\n            ORDER BY idx.indisunique DESC, i.relname;"
        cursor = self.execute_sql(query, (table, 'r', schema or 'public'))
        return [IndexMetadata(name, sql.rstrip(' ;'), columns.split(','), is_unique, table) for name, sql, is_unique, columns in cursor.fetchall()]

    def get_columns(self, table, schema=None):
        query = '\n            SELECT column_name, is_nullable, data_type, column_default\n            FROM information_schema.columns\n            WHERE table_name = %s AND table_schema = %s\n            ORDER BY ordinal_position'
        cursor = self.execute_sql(query, (table, schema or 'public'))
        pks = set(self.get_primary_keys(table, schema))
        return [ColumnMetadata(name, dt, null == 'YES', name in pks, table, df) for name, null, dt, df in cursor.fetchall()]

    def get_primary_keys(self, table, schema=None):
        query = '\n            SELECT kc.column_name\n            FROM information_schema.table_constraints AS tc\n            INNER JOIN information_schema.key_column_usage AS kc ON (\n                tc.table_name = kc.table_name AND\n                tc.table_schema = kc.table_schema AND\n                tc.constraint_name = kc.constraint_name)\n            WHERE\n                tc.constraint_type = %s AND\n                tc.table_name = %s AND\n                tc.table_schema = %s'
        ctype = 'PRIMARY KEY'
        cursor = self.execute_sql(query, (ctype, table, schema or 'public'))
        return [pk for pk, in cursor.fetchall()]

    def get_foreign_keys(self, table, schema=None):
        sql = "\n            SELECT DISTINCT\n                kcu.column_name, ccu.table_name, ccu.column_name\n            FROM information_schema.table_constraints AS tc\n            JOIN information_schema.key_column_usage AS kcu\n                ON (tc.constraint_name = kcu.constraint_name AND\n                    tc.constraint_schema = kcu.constraint_schema AND\n                    tc.table_name = kcu.table_name AND\n                    tc.table_schema = kcu.table_schema)\n            JOIN information_schema.constraint_column_usage AS ccu\n                ON (ccu.constraint_name = tc.constraint_name AND\n                    ccu.constraint_schema = tc.constraint_schema)\n            WHERE\n                tc.constraint_type = 'FOREIGN KEY' AND\n                tc.table_name = %s AND\n                tc.table_schema = %s"
        cursor = self.execute_sql(sql, (table, schema or 'public'))
        return [ForeignKeyMetadata(row[0], row[1], row[2], table) for row in cursor.fetchall()]

    def sequence_exists(self, sequence):
        res = self.execute_sql("\n            SELECT COUNT(*) FROM pg_class, pg_namespace\n            WHERE relkind='S'\n                AND pg_class.relnamespace = pg_namespace.oid\n                AND relname=%s", (sequence,))
        return bool(res.fetchone()[0])

    def get_binary_type(self):
        return psycopg2.Binary

    def conflict_statement(self, on_conflict, query):
        return

    def conflict_update(self, oc, query):
        action = oc._action.lower() if oc._action else ''
        if action in ('ignore', 'nothing'):
            parts = [SQL('ON CONFLICT')]
            if oc._conflict_target:
                parts.append(EnclosedNodeList([Entity(col) if isinstance(col, basestring) else col for col in oc._conflict_target]))
            parts.append(SQL('DO NOTHING'))
            return NodeList(parts)
        elif action and action != 'update':
            raise ValueError('The only supported actions for conflict resolution with Postgresql are "ignore" or "update".')
        elif not oc._update and (not oc._preserve):
            raise ValueError('If you are not performing any updates (or preserving any INSERTed values), then the conflict resolution action should be set to "IGNORE".')
        elif not (oc._conflict_target or oc._conflict_constraint):
            raise ValueError('Postgres requires that a conflict target be specified when doing an upsert.')
        return self._build_on_conflict_update(oc, query)

    def extract_date(self, date_part, date_field):
        return fn.EXTRACT(NodeList((date_part, SQL('FROM'), date_field)))

    def truncate_date(self, date_part, date_field):
        return fn.DATE_TRUNC(date_part, date_field)

    def to_timestamp(self, date_field):
        return self.extract_date('EPOCH', date_field)

    def from_timestamp(self, date_field):
        return fn.to_timestamp(date_field)

    def get_noop_select(self, ctx):
        return ctx.sql(Select().columns(SQL('0')).where(SQL('false')))

    def set_time_zone(self, timezone):
        self.execute_sql('set time zone "%s";' % timezone)