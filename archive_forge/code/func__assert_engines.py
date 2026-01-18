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
@contextlib.contextmanager
def _assert_engines(self):
    """produce a mock series of engine calls.

        These are expected to match engine-related calls established
        by the test subject.

        """
    writer_conn = SingletonConnection()
    writer_engine = SingletonEngine(writer_conn)
    if self.slave_uri:
        async_reader_conn = SingletonConnection()
        async_reader_engine = SingletonEngine(async_reader_conn)
    else:
        async_reader_conn = writer_conn
        async_reader_engine = writer_engine
    if self.synchronous_reader:
        reader_engine = writer_engine
    else:
        reader_engine = async_reader_engine
    engines = AssertDataSource(writer_engine, reader_engine, async_reader_engine)

    def create_engine(sql_connection, **kw):
        if sql_connection == self.engine_uri:
            return engines.writer
        elif sql_connection == self.slave_uri:
            return engines.async_reader
        else:
            assert False
    engine_factory = mock.Mock(side_effect=create_engine)
    engine_factory(sql_connection=self.engine_uri, **{k: mock.ANY for k in self.factory._engine_cfg.keys()})
    if self.slave_uri:
        engine_factory(sql_connection=self.slave_uri, **{k: mock.ANY for k in self.factory._engine_cfg.keys()})
    yield AssertDataSource(writer_engine, reader_engine, async_reader_engine)
    self.assertEqual(engine_factory.mock_calls, self.create_engine.mock_calls)
    for sym in [enginefacade._WRITER, enginefacade._READER, enginefacade._ASYNC_READER]:
        self.assertEqual(engines.element_for_writer(sym).mock_calls, self.engines.element_for_writer(sym).mock_calls)