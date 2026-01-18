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
@testtools.skipIf(not _postgres_exists(), 'postgres is not available')
class PostgresPersistenceTest(BackendPersistenceTestMixin, test.TestCase):

    def _init_db(self):
        engine = None
        try:
            db_uri = _get_connect_string('postgres', USER, PASSWD, database='postgres')
            engine = sa.create_engine(db_uri)
            with contextlib.closing(engine.connect()) as conn:
                conn.connection.set_isolation_level(0)
                conn.execute('CREATE DATABASE %s' % DATABASE)
                conn.connection.set_isolation_level(1)
        except Exception as e:
            raise Exception('Failed to initialize PostgreSQL db: %s' % e)
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
        return _get_connect_string('postgres', USER, PASSWD, database=DATABASE)

    def _remove_db(self):
        engine = None
        try:
            db_uri = _get_connect_string('postgres', USER, PASSWD, database='postgres')
            engine = sa.create_engine(db_uri)
            with contextlib.closing(engine.connect()) as conn:
                conn.connection.set_isolation_level(0)
                conn.execute('DROP DATABASE IF EXISTS %s' % DATABASE)
                conn.connection.set_isolation_level(1)
        except Exception as e:
            raise Exception('Failed to remove temporary database: %s' % e)
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass