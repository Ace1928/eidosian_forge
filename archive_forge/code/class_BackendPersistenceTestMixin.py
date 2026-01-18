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
class BackendPersistenceTestMixin(base.PersistenceTestMixin, metaclass=abc.ABCMeta):
    """Specifies a backend type and does required setup and teardown."""

    def _get_connection(self):
        return self.backend.get_connection()

    def test_entrypoint(self):
        with contextlib.closing(backends.fetch(self.db_conf)) as backend:
            with contextlib.closing(backend.get_connection()):
                pass

    @abc.abstractmethod
    def _init_db(self):
        """Sets up the database, and returns the uri to that database."""

    @abc.abstractmethod
    def _remove_db(self):
        """Cleans up by removing the database once the tests are done."""

    def setUp(self):
        super(BackendPersistenceTestMixin, self).setUp()
        self.backend = None
        try:
            self.db_uri = self._init_db()
            self.db_conf = {'connection': self.db_uri}
            self.addCleanup(self._remove_db)
        except Exception as e:
            self.skipTest('Failed to create temporary database; testing being skipped due to: %s' % e)
        else:
            self.backend = impl_sqlalchemy.SQLAlchemyBackend(self.db_conf)
            self.addCleanup(self.backend.close)
            with contextlib.closing(self._get_connection()) as conn:
                conn.upgrade()