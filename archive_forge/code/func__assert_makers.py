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
def _assert_makers(self, engines):
    writer_session = mock.Mock(connection=mock.Mock(return_value=engines.writer._assert_connection))
    writer_maker = mock.Mock(return_value=writer_session)
    if self.slave_uri:
        async_reader_session = mock.Mock(connection=mock.Mock(return_value=engines.async_reader._assert_connection))
        async_reader_maker = mock.Mock(return_value=async_reader_session)
    else:
        async_reader_session = writer_session
        async_reader_maker = writer_maker
    if self.synchronous_reader:
        reader_maker = writer_maker
    else:
        reader_maker = async_reader_maker
    makers = AssertDataSource(writer_maker, reader_maker, async_reader_maker)

    def get_maker(engine, **kw):
        if engine is engines.writer:
            return makers.writer
        elif engine is engines.reader:
            return makers.reader
        elif engine is engines.async_reader:
            return makers.async_reader
        else:
            assert False
    maker_factories = mock.Mock(side_effect=get_maker)
    maker_factories(engine=engines.writer, expire_on_commit=False)
    if self.slave_uri:
        maker_factories(engine=engines.async_reader, expire_on_commit=False)
    yield makers
    self.assertEqual(maker_factories.mock_calls, self.get_maker.mock_calls)
    for sym in [enginefacade._WRITER, enginefacade._READER, enginefacade._ASYNC_READER]:
        self.assertEqual(makers.element_for_writer(sym).mock_calls, self.makers.element_for_writer(sym).mock_calls)