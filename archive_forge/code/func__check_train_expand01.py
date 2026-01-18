import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_train_expand01(self, engine, data):
    images = db_utils.get_table(engine, 'images')
    with engine.connect() as conn:
        rows = conn.execute(images.select().order_by(images.c.id)).fetchall()
    self.assertEqual(0, len(rows))
    data_migrations.migrate(engine)