from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
class TablesTest(TestBase):
    run_setup_bind = 'once'
    run_define_tables = 'once'
    run_create_tables = 'once'
    run_inserts = 'each'
    run_deletes = 'each'
    run_dispose_bind = None
    bind = None
    _tables_metadata = None
    tables = None
    other = None
    sequences = None

    @config.fixture(autouse=True, scope='class')
    def _setup_tables_test_class(self):
        cls = self.__class__
        cls._init_class()
        cls._setup_once_tables()
        cls._setup_once_inserts()
        yield
        cls._teardown_once_metadata_bind()

    @config.fixture(autouse=True, scope='function')
    def _setup_tables_test_instance(self):
        self._setup_each_tables()
        self._setup_each_inserts()
        yield
        self._teardown_each_tables()

    @property
    def tables_test_metadata(self):
        return self._tables_metadata

    @classmethod
    def _init_class(cls):
        if cls.run_define_tables == 'each':
            if cls.run_create_tables == 'once':
                cls.run_create_tables = 'each'
            assert cls.run_inserts in ('each', None)
        cls.other = adict()
        cls.tables = adict()
        cls.sequences = adict()
        cls.bind = cls.setup_bind()
        cls._tables_metadata = sa.MetaData()

    @classmethod
    def _setup_once_inserts(cls):
        if cls.run_inserts == 'once':
            cls._load_fixtures()
            with cls.bind.begin() as conn:
                cls.insert_data(conn)

    @classmethod
    def _setup_once_tables(cls):
        if cls.run_define_tables == 'once':
            cls.define_tables(cls._tables_metadata)
            if cls.run_create_tables == 'once':
                cls._tables_metadata.create_all(cls.bind)
            cls.tables.update(cls._tables_metadata.tables)
            cls.sequences.update(cls._tables_metadata._sequences)

    def _setup_each_tables(self):
        if self.run_define_tables == 'each':
            self.define_tables(self._tables_metadata)
            if self.run_create_tables == 'each':
                self._tables_metadata.create_all(self.bind)
            self.tables.update(self._tables_metadata.tables)
            self.sequences.update(self._tables_metadata._sequences)
        elif self.run_create_tables == 'each':
            self._tables_metadata.create_all(self.bind)

    def _setup_each_inserts(self):
        if self.run_inserts == 'each':
            self._load_fixtures()
            with self.bind.begin() as conn:
                self.insert_data(conn)

    def _teardown_each_tables(self):
        if self.run_define_tables == 'each':
            self.tables.clear()
            if self.run_create_tables == 'each':
                drop_all_tables_from_metadata(self._tables_metadata, self.bind)
            self._tables_metadata.clear()
        elif self.run_create_tables == 'each':
            drop_all_tables_from_metadata(self._tables_metadata, self.bind)
        savepoints = getattr(config.requirements, 'savepoints', False)
        if savepoints:
            savepoints = savepoints.enabled
        if self.run_define_tables != 'each' and self.run_create_tables != 'each' and (self.run_deletes == 'each'):
            with self.bind.begin() as conn:
                for table in reversed([t for t, fks in sort_tables_and_constraints(self._tables_metadata.tables.values()) if t is not None]):
                    try:
                        if savepoints:
                            with conn.begin_nested():
                                conn.execute(table.delete())
                        else:
                            conn.execute(table.delete())
                    except sa.exc.DBAPIError as ex:
                        print('Error emptying table %s: %r' % (table, ex), file=sys.stderr)

    @classmethod
    def _teardown_once_metadata_bind(cls):
        if cls.run_create_tables:
            drop_all_tables_from_metadata(cls._tables_metadata, cls.bind)
        if cls.run_dispose_bind == 'once':
            cls.dispose_bind(cls.bind)
        cls._tables_metadata.bind = None
        if cls.run_setup_bind is not None:
            cls.bind = None

    @classmethod
    def setup_bind(cls):
        return config.db

    @classmethod
    def dispose_bind(cls, bind):
        if hasattr(bind, 'dispose'):
            bind.dispose()
        elif hasattr(bind, 'close'):
            bind.close()

    @classmethod
    def define_tables(cls, metadata):
        pass

    @classmethod
    def fixtures(cls):
        return {}

    @classmethod
    def insert_data(cls, connection):
        pass

    def sql_count_(self, count, fn):
        self.assert_sql_count(self.bind, fn, count)

    def sql_eq_(self, callable_, statements):
        self.assert_sql(self.bind, callable_, statements)

    @classmethod
    def _load_fixtures(cls):
        """Insert rows as represented by the fixtures() method."""
        headers, rows = ({}, {})
        for table, data in cls.fixtures().items():
            if len(data) < 2:
                continue
            if isinstance(table, str):
                table = cls.tables[table]
            headers[table] = data[0]
            rows[table] = data[1:]
        for table, fks in sort_tables_and_constraints(cls._tables_metadata.tables.values()):
            if table is None:
                continue
            if table not in headers:
                continue
            with cls.bind.begin() as conn:
                conn.execute(table.insert(), [dict(zip(headers[table], column_values)) for column_values in rows[table]])