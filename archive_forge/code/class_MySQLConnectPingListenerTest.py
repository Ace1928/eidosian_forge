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
class MySQLConnectPingListenerTest(db_test_base._MySQLOpportunisticTestCase):

    def test__connect_ping_listener(self):
        for idx in range(2):
            with self.engine.begin() as conn:
                self.assertIsInstance(conn._transaction, base_engine.RootTransaction)
                if compat.sqla_2:
                    engines._connect_ping_listener(conn)
                    self.assertIsNone(conn._transaction)
                else:
                    engines._connect_ping_listener(conn, False)
                    self.assertIsInstance(conn._transaction, base_engine.RootTransaction)