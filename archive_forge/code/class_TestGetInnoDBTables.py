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
class TestGetInnoDBTables(db_test_base._MySQLOpportunisticTestCase):

    def test_all_tables_use_innodb(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE customers (a INT, b CHAR (20), INDEX (a)) ENGINE=InnoDB'))
        self.assertEqual([], utils.get_non_innodb_tables(self.engine))

    def test_all_tables_use_innodb_false(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE employee (i INT) ENGINE=MEMORY'))
        self.assertEqual(['employee'], utils.get_non_innodb_tables(self.engine))

    def test_skip_tables_use_default_value(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE migrate_version (i INT) ENGINE=MEMORY'))
        self.assertEqual([], utils.get_non_innodb_tables(self.engine))

    def test_skip_tables_use_passed_value(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE some_table (i INT) ENGINE=MEMORY'))
        self.assertEqual([], utils.get_non_innodb_tables(self.engine, skip_tables=('some_table',)))

    def test_skip_tables_use_empty_list(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE some_table_3 (i INT) ENGINE=MEMORY'))
        self.assertEqual(['some_table_3'], utils.get_non_innodb_tables(self.engine, skip_tables=()))

    def test_skip_tables_use_several_values(self):
        with self.engine.connect() as conn, conn.begin():
            conn.execute(sql.text('CREATE TABLE some_table_1 (i INT) ENGINE=MEMORY'))
            conn.execute(sql.text('CREATE TABLE some_table_2 (i INT) ENGINE=MEMORY'))
        self.assertEqual([], utils.get_non_innodb_tables(self.engine, skip_tables=('some_table_1', 'some_table_2')))