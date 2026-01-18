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
@testtools.skipIf(not _mysql_exists(), 'mysql is not available')
class MysqlPersistenceTest(BackendPersistenceTestMixin, test.TestCase):

    def _init_db(self):
        engine = None
        try:
            db_uri = _get_connect_string('mysql', USER, PASSWD)
            engine = sa.create_engine(db_uri)
            with contextlib.closing(engine.connect()) as conn:
                conn.execute('CREATE DATABASE %s' % DATABASE)
        except Exception as e:
            raise Exception('Failed to initialize MySQL db: %s' % e)
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
        return _get_connect_string('mysql', USER, PASSWD, database=DATABASE)

    def _remove_db(self):
        engine = None
        try:
            engine = sa.create_engine(self.db_uri)
            with contextlib.closing(engine.connect()) as conn:
                conn.execute('DROP DATABASE IF EXISTS %s' % DATABASE)
        except Exception as e:
            raise Exception('Failed to remove temporary database: %s' % e)
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass