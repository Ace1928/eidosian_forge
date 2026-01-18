import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestOcataMigrate01Mixin(test_migrations.AlembicMigrationsMixin):

    def _get_revisions(self, config):
        return test_migrations.AlembicMigrationsMixin._get_revisions(self, config, head='ocata_expand01')

    def _pre_upgrade_ocata_expand01(self, engine):
        images = db_utils.get_table(engine, 'images')
        image_members = db_utils.get_table(engine, 'image_members')
        now = datetime.datetime.now()
        public_temp = dict(deleted=False, created_at=now, status='active', is_public=True, min_disk=0, min_ram=0, id='public_id')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(public_temp))
        shared_temp = dict(deleted=False, created_at=now, status='active', is_public=False, min_disk=0, min_ram=0, id='shared_id')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(shared_temp))
        private_temp = dict(deleted=False, created_at=now, status='active', is_public=False, min_disk=0, min_ram=0, id='private_id_1')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(private_temp))
        private_temp = dict(deleted=False, created_at=now, status='active', is_public=False, min_disk=0, min_ram=0, id='private_id_2')
        with engine.connect() as conn, conn.begin():
            conn.execute(images.insert().values(private_temp))
        temp = dict(deleted=False, created_at=now, image_id='shared_id', member='fake_member_452', can_share=True, id=45)
        with engine.connect() as conn, conn.begin():
            conn.execute(image_members.insert().values(temp))
        temp = dict(deleted=True, created_at=now, image_id='shared_id', member='fake_member_453', can_share=True, id=453)
        with engine.connect() as conn, conn.begin():
            conn.execute(image_members.insert().values(temp))
        temp = dict(deleted=True, created_at=now, image_id='private_id_2', member='fake_member_451', can_share=True, id=451)
        with engine.connect() as conn, conn.begin():
            conn.execute(image_members.insert().values(temp))
        temp = dict(deleted=False, created_at=now, image_id='public_id', member='fake_member_450', can_share=True, id=450)
        with engine.connect() as conn, conn.begin():
            conn.execute(image_members.insert().values(temp))

    def _check_ocata_expand01(self, engine, data):
        images = db_utils.get_table(engine, 'images')
        with engine.connect() as conn:
            rows = conn.execute(images.select().order_by(images.c.id)).fetchall()
        self.assertEqual(4, len(rows))
        for row in rows:
            self.assertIsNone(row.visibility)
        data_migrations.migrate(engine)
        with engine.connect() as conn:
            rows = conn.execute(images.select().order_by(images.c.id)).fetchall()
        self.assertEqual(4, len(rows))
        self.assertEqual('private_id_1', rows[0].id)
        self.assertEqual('private_id_2', rows[1].id)
        self.assertEqual('public_id', rows[2].id)
        self.assertEqual('shared_id', rows[3].id)