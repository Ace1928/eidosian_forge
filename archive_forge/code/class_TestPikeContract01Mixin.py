from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestPikeContract01Mixin(test_migrations.AlembicMigrationsMixin):
    artifacts_table_names = ['artifact_blob_locations', 'artifact_properties', 'artifact_blobs', 'artifact_dependencies', 'artifact_tags', 'artifacts']

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='pike_contract01')

    def _pre_upgrade_pike_contract01(self, engine):
        for table_name in self.artifacts_table_names:
            table = db_utils.get_table(engine, table_name)
            self.assertIsNotNone(table)

    def _check_pike_contract01(self, engine, data):
        for table_name in self.artifacts_table_names:
            self.assertRaises(sqlalchemy.exc.NoSuchTableError, db_utils.get_table, engine, table_name)