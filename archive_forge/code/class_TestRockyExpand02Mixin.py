from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestRockyExpand02Mixin(test_migrations.AlembicMigrationsMixin):

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='rocky_expand02')

    def _pre_upgrade_rocky_expand02(self, engine):
        images = db_utils.get_table(engine, 'images')
        self.assertNotIn('os_hash_algo', images.c)
        self.assertNotIn('os_hash_value', images.c)

    def _check_rocky_expand02(self, engine, data):
        images = db_utils.get_table(engine, 'images')
        self.assertIn('os_hash_algo', images.c)
        self.assertTrue(images.c.os_hash_algo.nullable)
        self.assertIn('os_hash_value', images.c)
        self.assertTrue(images.c.os_hash_value.nullable)