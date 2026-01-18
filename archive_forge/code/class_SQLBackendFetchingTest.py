import abc
import contextlib
import os
import random
import tempfile
import testtools
import sqlalchemy as sa
from taskflow.persistence import backends
from taskflow.persistence.backends import impl_sqlalchemy
from taskflow import test
from taskflow.tests.unit.persistence import base
class SQLBackendFetchingTest(test.TestCase):

    def test_sqlite_persistence_entry_point(self):
        conf = {'connection': 'sqlite:///'}
        with contextlib.closing(backends.fetch(conf)) as be:
            self.assertIsInstance(be, impl_sqlalchemy.SQLAlchemyBackend)

    @testtools.skipIf(not _mysql_exists(), 'mysql is not available')
    def test_mysql_persistence_entry_point(self):
        uri = _get_connect_string('mysql', USER, PASSWD, database=DATABASE)
        conf = {'connection': uri}
        with contextlib.closing(backends.fetch(conf)) as be:
            self.assertIsInstance(be, impl_sqlalchemy.SQLAlchemyBackend)

    @testtools.skipIf(not _postgres_exists(), 'postgres is not available')
    def test_postgres_persistence_entry_point(self):
        uri = _get_connect_string('postgres', USER, PASSWD, database=DATABASE)
        conf = {'connection': uri}
        with contextlib.closing(backends.fetch(conf)) as be:
            self.assertIsInstance(be, impl_sqlalchemy.SQLAlchemyBackend)