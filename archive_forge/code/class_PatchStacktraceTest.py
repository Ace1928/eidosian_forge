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
class PatchStacktraceTest(db_test_base._DbTestCase):

    def test_trace(self):
        engine = self.engine
        import traceback
        orig_extract_stack = traceback.extract_stack

        def extract_stack():
            return [(row[0].replace('oslo_db/', ''), row[1], row[2], row[3]) for row in orig_extract_stack()]
        with mock.patch('traceback.extract_stack', side_effect=extract_stack):
            engines._add_trace_comments(engine)
            conn = engine.connect()
            orig_do_exec = engine.dialect.do_execute
            with mock.patch.object(engine.dialect, 'do_execute') as mock_exec:
                mock_exec.side_effect = orig_do_exec
                conn.execute(sql.text('select 1'))
            call = mock_exec.mock_calls[0]
            caller = os.path.join('tests', 'sqlalchemy', 'test_sqlalchemy.py')
            self.assertIn(caller, call[1][1])