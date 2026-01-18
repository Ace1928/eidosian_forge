from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class TestMigrationUtils(db_test_base._DbTestCase):
    """Class for testing utils that are used in db migrations."""

    def setUp(self):
        super(TestMigrationUtils, self).setUp()
        self.meta = MetaData()
        self.conn = self.engine.connect()
        self.addCleanup(self.meta.drop_all, self.engine)
        self.addCleanup(self.conn.close)

    def _populate_db_for_drop_duplicate_entries(self, engine, meta, table_name):
        values = [{'id': 11, 'a': 3, 'b': 10, 'c': 'abcdef'}, {'id': 12, 'a': 5, 'b': 10, 'c': 'abcdef'}, {'id': 13, 'a': 6, 'b': 10, 'c': 'abcdef'}, {'id': 14, 'a': 7, 'b': 10, 'c': 'abcdef'}, {'id': 21, 'a': 1, 'b': 20, 'c': 'aa'}, {'id': 31, 'a': 1, 'b': 20, 'c': 'bb'}, {'id': 41, 'a': 1, 'b': 30, 'c': 'aef'}, {'id': 42, 'a': 2, 'b': 30, 'c': 'aef'}, {'id': 43, 'a': 3, 'b': 30, 'c': 'aef'}]
        test_table = Table(table_name, meta, Column('id', Integer, primary_key=True, nullable=False), Column('a', Integer), Column('b', Integer), Column('c', String(255)), Column('deleted', Integer, default=0), Column('deleted_at', DateTime), Column('updated_at', DateTime))
        test_table.create(engine)
        with engine.connect() as conn, conn.begin():
            conn.execute(test_table.insert(), values)
        return (test_table, values)

    def test_drop_old_duplicate_entries_from_table(self):
        table_name = '__test_tmp_table__'
        test_table, values = self._populate_db_for_drop_duplicate_entries(self.engine, self.meta, table_name)
        utils.drop_old_duplicate_entries_from_table(self.engine, table_name, False, 'b', 'c')
        uniq_values = set()
        expected_ids = []
        for value in sorted(values, key=lambda x: x['id'], reverse=True):
            uniq_value = (('b', value['b']), ('c', value['c']))
            if uniq_value in uniq_values:
                continue
            uniq_values.add(uniq_value)
            expected_ids.append(value['id'])
        with self.engine.connect() as conn, conn.begin():
            real_ids = [row[0] for row in conn.execute(select(test_table.c.id)).fetchall()]
        self.assertEqual(len(expected_ids), len(real_ids))
        for id_ in expected_ids:
            self.assertIn(id_, real_ids)

    def test_drop_dup_entries_in_file_conn(self):
        table_name = '__test_tmp_table__'
        tmp_db_file = self.create_tempfiles([['name', '']], ext='.sql')[0]
        in_file_engine = session.EngineFacade('sqlite:///%s' % tmp_db_file).get_engine()
        meta = MetaData()
        test_table, values = self._populate_db_for_drop_duplicate_entries(in_file_engine, meta, table_name)
        utils.drop_old_duplicate_entries_from_table(in_file_engine, table_name, False, 'b', 'c')

    def test_drop_old_duplicate_entries_from_table_soft_delete(self):
        table_name = '__test_tmp_table__'
        table, values = self._populate_db_for_drop_duplicate_entries(self.engine, self.meta, table_name)
        utils.drop_old_duplicate_entries_from_table(self.engine, table_name, True, 'b', 'c')
        uniq_values = set()
        expected_values = []
        soft_deleted_values = []
        for value in sorted(values, key=lambda x: x['id'], reverse=True):
            uniq_value = (('b', value['b']), ('c', value['c']))
            if uniq_value in uniq_values:
                soft_deleted_values.append(value)
                continue
            uniq_values.add(uniq_value)
            expected_values.append(value)
        base_select = table.select()
        with self.engine.connect() as conn, conn.begin():
            rows_select = base_select.where(table.c.deleted != table.c.id)
            row_ids = [row.id for row in conn.execute(rows_select).fetchall()]
            self.assertEqual(len(expected_values), len(row_ids))
            for value in expected_values:
                self.assertIn(value['id'], row_ids)
            deleted_rows_select = base_select.where(table.c.deleted == table.c.id)
            deleted_rows_ids = [row.id for row in conn.execute(deleted_rows_select).fetchall()]
        self.assertEqual(len(values) - len(row_ids), len(deleted_rows_ids))
        for value in soft_deleted_values:
            self.assertIn(value['id'], deleted_rows_ids)

    def test_get_foreign_key_constraint_name(self):
        table_1 = Table('table_name_1', self.meta, Column('id', Integer, primary_key=True), Column('deleted', Integer))
        table_2 = Table('table_name_2', self.meta, Column('id', Integer, primary_key=True), Column('foreign_id', Integer), ForeignKeyConstraint(['foreign_id'], ['table_name_1.id'], name='table_name_2_fk1'), Column('deleted', Integer))
        self.meta.create_all(self.engine, tables=[table_1, table_2])
        fkc = utils.get_foreign_key_constraint_name(self.engine, 'table_name_2', 'foreign_id')
        self.assertEqual(fkc, 'table_name_2_fk1')