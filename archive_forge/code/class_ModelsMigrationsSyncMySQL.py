from unittest import mock
import sqlalchemy as sa
from sqlalchemy import orm
from oslo_db.sqlalchemy import test_migrations as migrate
from oslo_db.tests.sqlalchemy import base as db_test_base
class ModelsMigrationsSyncMySQL(ModelsMigrationSyncMixin, migrate.ModelsMigrationsSync, db_test_base._MySQLOpportunisticTestCase):

    def test_models_not_sync(self):
        self._test_models_not_sync()

    def test_models_not_sync_filtered(self):
        self._test_models_not_sync_filtered()