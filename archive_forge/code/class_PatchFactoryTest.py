import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
class PatchFactoryTest(test_base.BaseTestCase):

    def test_patch_manager(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite:///foo.db')
        alt_mgr = enginefacade.transaction_context()
        alt_mgr.configure(connection='sqlite:///bar.db')

        @normal_mgr.writer
        def go1(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///foo.db')
            self.assertIs(s1.bind, normal_mgr._factory._writer_engine)

        @normal_mgr.writer
        def go2(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///bar.db')
            self.assertIs(normal_mgr._factory._writer_engine, alt_mgr._factory._writer_engine)

        def create_engine(sql_connection, **kw):
            return mock.Mock(url=sql_connection)
        with mock.patch('oslo_db.sqlalchemy.engines.create_engine', create_engine):
            context = oslo_context.RequestContext()
            go1(context)
            reset = normal_mgr.patch_factory(alt_mgr)
            go2(context)
            reset()
            go1(context)

    def test_patch_factory(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite:///foo.db')
        alt_mgr = enginefacade.transaction_context()
        alt_mgr.configure(connection='sqlite:///bar.db')

        @normal_mgr.writer
        def go1(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///foo.db')
            self.assertIs(s1.bind, normal_mgr._factory._writer_engine)

        @normal_mgr.writer
        def go2(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///bar.db')
            self.assertIs(normal_mgr._factory._writer_engine, alt_mgr._factory._writer_engine)

        def create_engine(sql_connection, **kw):
            return mock.Mock(url=sql_connection)
        with mock.patch('oslo_db.sqlalchemy.engines.create_engine', create_engine):
            context = oslo_context.RequestContext()
            go1(context)
            reset = normal_mgr.patch_factory(alt_mgr._factory)
            go2(context)
            reset()
            go1(context)

    def test_patch_engine(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite:///foo.db', rollback_reader_sessions=True)

        @normal_mgr.writer
        def go1(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///foo.db')
            self.assertIs(s1.bind, normal_mgr._factory._writer_engine)

        @normal_mgr.writer
        def go2(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///bar.db')
            self.assertTrue(enginefacade._transaction_ctx_for_context(context).rollback_reader_sessions)
            self.assertTrue(enginefacade._transaction_ctx_for_context(context).factory.synchronous_reader)

        def create_engine(sql_connection, **kw):
            return mock.Mock(url=sql_connection)
        with mock.patch('oslo_db.sqlalchemy.engines.create_engine', create_engine):
            mock_engine = create_engine('sqlite:///bar.db')
            context = oslo_context.RequestContext()
            go1(context)
            reset = normal_mgr.patch_engine(mock_engine)
            go2(context)
            self.assertIs(normal_mgr._factory._writer_engine, mock_engine)
            reset()
            go1(context)

    def test_patch_not_started(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite:///foo.db', rollback_reader_sessions=True)

        @normal_mgr.writer
        def go1(context):
            s1 = context.session
            self.assertEqual(s1.bind.url, 'sqlite:///bar.db')
            self.assertTrue(enginefacade._transaction_ctx_for_context(context).rollback_reader_sessions)

        def create_engine(sql_connection, **kw):
            return mock.Mock(url=sql_connection)
        with mock.patch('oslo_db.sqlalchemy.engines.create_engine', create_engine):
            mock_engine = create_engine('sqlite:///bar.db')
            context = oslo_context.RequestContext()
            reset = normal_mgr.patch_engine(mock_engine)
            go1(context)
            self.assertIs(normal_mgr._factory._writer_engine, mock_engine)
            reset()

    def test_new_manager_from_config(self):
        normal_mgr = enginefacade.transaction_context()
        normal_mgr.configure(connection='sqlite://', sqlite_fk=True, mysql_sql_mode='FOOBAR', max_overflow=38)
        normal_mgr._factory._start()
        copied_mgr = normal_mgr.make_new_manager()
        self.assertTrue(normal_mgr._factory._started)
        self.assertIsNotNone(normal_mgr._factory._writer_engine)
        self.assertIsNot(copied_mgr._factory, normal_mgr._factory)
        self.assertFalse(copied_mgr._factory._started)
        copied_mgr._factory._start()
        self.assertIsNot(normal_mgr._factory._writer_engine, copied_mgr._factory._writer_engine)
        engine_args = copied_mgr._factory._engine_args_for_conf(None)
        self.assertTrue(engine_args['sqlite_fk'])
        self.assertEqual('FOOBAR', engine_args['mysql_sql_mode'])
        self.assertEqual(38, engine_args['max_overflow'])
        self.assertNotIn('mysql_wsrep_sync_wait', engine_args)

    def test_new_manager_from_options(self):
        """test enginefacade's defaults given a default structure from opts"""
        factory = enginefacade._TransactionFactory()
        cfg.CONF.register_opts(options.database_opts, 'database')
        factory.configure(**dict(cfg.CONF.database.items()))
        engine_args = factory._engine_args_for_conf(None)
        self.assertEqual(None, engine_args['mysql_wsrep_sync_wait'])
        self.assertEqual(True, engine_args['sqlite_synchronous'])
        self.assertEqual('TRADITIONAL', engine_args['mysql_sql_mode'])