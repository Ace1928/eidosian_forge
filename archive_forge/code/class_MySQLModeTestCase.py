import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class MySQLModeTestCase(db_test_base._MySQLOpportunisticTestCase):

    def __init__(self, *args, **kwargs):
        super(MySQLModeTestCase, self).__init__(*args, **kwargs)
        self.mysql_mode = ''

    def setUp(self):
        super(MySQLModeTestCase, self).setUp()
        mode_engine = session.create_engine(self.engine.url, mysql_sql_mode=self.mysql_mode)
        self.connection = mode_engine.connect()
        meta = MetaData()
        self.test_table = Table(_TABLE_NAME + 'mode', meta, Column('id', Integer, primary_key=True), Column('bar', String(255)))
        with self.connection.begin():
            self.test_table.create(self.connection)

        def cleanup():
            with self.connection.begin():
                self.test_table.drop(self.connection)
            self.connection.close()
            mode_engine.dispose()
        self.addCleanup(cleanup)

    def _test_string_too_long(self, value):
        with self.connection.begin():
            self.connection.execute(self.test_table.insert(), {'bar': value})
            result = self.connection.execute(self.test_table.select())
            return result.fetchone().bar

    def test_string_too_long(self):
        value = 'a' * 512
        self.assertNotEqual(value, self._test_string_too_long(value))