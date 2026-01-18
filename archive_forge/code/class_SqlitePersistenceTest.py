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
class SqlitePersistenceTest(test.TestCase, base.PersistenceTestMixin):
    """Inherits from the base test and sets up a sqlite temporary db."""

    def _get_connection(self):
        conf = {'connection': self.db_uri}
        return impl_sqlalchemy.SQLAlchemyBackend(conf).get_connection()

    def setUp(self):
        super(SqlitePersistenceTest, self).setUp()
        self.db_location = tempfile.mktemp(suffix='.db')
        self.db_uri = 'sqlite:///%s' % self.db_location
        with contextlib.closing(self._get_connection()) as conn:
            conn.upgrade()

    def tearDown(self):
        super(SqlitePersistenceTest, self).tearDown()
        if self.db_location and os.path.isfile(self.db_location):
            os.unlink(self.db_location)
            self.db_location = None