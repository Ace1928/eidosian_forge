import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestOcataContract01Mixin(test_migrations.AlembicMigrationsMixin):

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='ocata_contract01')

    def _pre_upgrade_ocata_contract01(self, engine):
        images = db_utils.get_table(engine, 'images')
        now = datetime.datetime.now()
        self.assertIn('is_public', images.c)
        self.assertIn('visibility', images.c)
        self.assertTrue(images.c.is_public.nullable)
        self.assertTrue(images.c.visibility.nullable)
        public_temp = dict(deleted=False, created_at=now, status='active', is_public=True, min_disk=0, min_ram=0, id='public_id_before_expand')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(public_temp))
        shared_temp = dict(deleted=False, created_at=now, status='active', is_public=False, min_disk=0, min_ram=0, id='private_id_before_expand')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(shared_temp))
        data_migrations.migrate(engine=engine, release='ocata')

    def _check_ocata_contract01(self, engine, data):
        images = db_utils.get_table(engine, 'images')
        self.assertNotIn('is_public', images.c)
        self.assertIn('visibility', images.c)