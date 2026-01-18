from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _pre_upgrade_wallaby_expand01(self, engine):
    tasks = db_utils.get_table(engine, 'tasks')
    self.assertNotIn('image_id', tasks.c)
    self.assertNotIn('request_id', tasks.c)
    self.assertNotIn('user_id', tasks.c)
    self.assertFalse(db_utils.index_exists(engine, 'tasks', 'ix_tasks_image_id'))