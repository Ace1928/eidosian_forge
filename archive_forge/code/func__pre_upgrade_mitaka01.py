from oslo_db.sqlalchemy import test_fixtures
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _pre_upgrade_mitaka01(self, engine):
    indexes = get_indexes('images', engine)
    self.assertNotIn('created_at_image_idx', indexes)
    self.assertNotIn('updated_at_image_idx', indexes)