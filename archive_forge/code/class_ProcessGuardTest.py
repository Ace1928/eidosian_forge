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
class ProcessGuardTest(db_test_base._DbTestCase):

    def test_process_guard(self):
        self.engine.dispose()

        def get_parent_pid():
            return 4

        def get_child_pid():
            return 5
        with mock.patch('os.getpid', get_parent_pid):
            with self.engine.connect() as conn:
                dbapi_id = id(compat.driver_connection(conn))
        with mock.patch('os.getpid', get_child_pid):
            with self.engine.connect() as conn:
                new_dbapi_id = id(compat.driver_connection(conn))
        self.assertNotEqual(dbapi_id, new_dbapi_id)
        with mock.patch('os.getpid', get_child_pid):
            with self.engine.connect() as conn:
                newer_dbapi_id = id(compat.driver_connection(conn))
        self.assertEqual(new_dbapi_id, newer_dbapi_id)