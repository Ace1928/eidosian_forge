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
class CreateEngineTest(test_base.BaseTestCase):
    """Test that dialect-specific arguments/ listeners are set up correctly.

    """

    def setUp(self):
        super(CreateEngineTest, self).setUp()
        self.args = {'connect_args': {}}

    def test_queuepool_args(self):
        engines._init_connection_args(utils.make_url('mysql+pymysql://u:p@host/test'), self.args, {'max_pool_size': 10, 'max_overflow': 10})
        self.assertEqual(10, self.args['pool_size'])
        self.assertEqual(10, self.args['max_overflow'])

    def test_sqlite_memory_pool_args(self):
        for _url in ('sqlite://', 'sqlite:///:memory:'):
            engines._init_connection_args(utils.make_url(_url), self.args, {'max_pool_size': 10, 'max_overflow': 10})
            self.assertNotIn('pool_size', self.args)
            self.assertNotIn('max_overflow', self.args)
            self.assertEqual(False, self.args['connect_args']['check_same_thread'])
            self.assertIn('poolclass', self.args)

    def test_sqlite_file_pool_args(self):
        engines._init_connection_args(utils.make_url('sqlite:///somefile.db'), self.args, {'max_pool_size': 10, 'max_overflow': 10})
        self.assertNotIn('pool_size', self.args)
        self.assertNotIn('max_overflow', self.args)
        self.assertFalse(self.args['connect_args'])
        if not compat.sqla_2:
            self.assertNotIn('poolclass', self.args)
        else:
            self.assertIs(self.args['poolclass'], NullPool)

    def _test_mysql_connect_args_default(self, connect_args):
        self.assertEqual({'charset': 'utf8', 'use_unicode': 1}, connect_args)

    def test_mysql_connect_args_default(self):
        engines._init_connection_args(utils.make_url('mysql://u:p@host/test'), self.args, {})
        self._test_mysql_connect_args_default(self.args['connect_args'])

    def test_mysql_pymysql_connect_args_default(self):
        engines._init_connection_args(utils.make_url('mysql+pymysql://u:p@host/test'), self.args, {})
        self.assertEqual({'charset': 'utf8'}, self.args['connect_args'])

    def test_mysql_mysqldb_connect_args_default(self):
        engines._init_connection_args(utils.make_url('mysql+mysqldb://u:p@host/test'), self.args, {})
        self._test_mysql_connect_args_default(self.args['connect_args'])

    def test_postgresql_connect_args_default(self):
        engines._init_connection_args(utils.make_url('postgresql://u:p@host/test'), self.args, {})
        self.assertEqual('utf8', self.args['client_encoding'])
        self.assertFalse(self.args['connect_args'])

    def test_mysqlconnector_raise_on_warnings_default(self):
        engines._init_connection_args(utils.make_url('mysql+mysqlconnector://u:p@host/test'), self.args, {})
        self.assertEqual(False, self.args['connect_args']['raise_on_warnings'])

    def test_mysqlconnector_raise_on_warnings_override(self):
        engines._init_connection_args(utils.make_url('mysql+mysqlconnector://u:p@host/test?raise_on_warnings=true'), self.args, {})
        self.assertNotIn('raise_on_warnings', self.args['connect_args'])

    def test_thread_checkin(self):
        with mock.patch('sqlalchemy.event.listens_for'):
            with mock.patch('sqlalchemy.event.listen') as listen_evt:
                engines._init_events.dispatch_on_drivername('sqlite')(mock.Mock())
        self.assertEqual(listen_evt.mock_calls[0][1][-1], engines._thread_yield)

    def test_warn_on_missing_driver(self):
        warnings = mock.Mock()

        def warn_interpolate(msg, args):
            warnings.warning(msg % (args,))
        with mock.patch('oslo_db.sqlalchemy.engines.LOG.warning', warn_interpolate):
            engines._vet_url(utils.make_url('mysql://scott:tiger@some_host/some_db'))
            engines._vet_url(utils.make_url('mysql+mysqldb://scott:tiger@some_host/some_db'))
            engines._vet_url(utils.make_url('mysql+pymysql://scott:tiger@some_host/some_db'))
            engines._vet_url(utils.make_url('postgresql+psycopg2://scott:tiger@some_host/some_db'))
            engines._vet_url(utils.make_url('postgresql://scott:tiger@some_host/some_db'))
        self.assertEqual([mock.call.warning("URL mysql://scott:***@some_host/some_db does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended. For MySQL, it is strongly recommended that mysql+pymysql:// be specified for maximum service compatibility"), mock.call.warning("URL postgresql://scott:***@some_host/some_db does not contain a '+drivername' portion, and will make use of a default driver.  A full dbname+drivername:// protocol is recommended.")], warnings.mock_calls)