import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _test_trigger_new_to_old(self, engine, images):
    now = datetime.datetime.now()
    public_temp = dict(deleted=False, created_at=now, status='active', visibility='public', min_disk=0, min_ram=0, id='public_id_new_to_old')
    with engine.connect() as conn, conn.begin():
        conn.execute(images.insert().values(public_temp))
    shared_temp = dict(deleted=False, created_at=now, status='active', visibility='private', min_disk=0, min_ram=0, id='private_id_new_to_old')
    with engine.connect() as conn, conn.begin():
        conn.execute(images.insert().values(shared_temp))
    shared_temp = dict(deleted=False, created_at=now, status='active', visibility='shared', min_disk=0, min_ram=0, id='shared_id_new_to_old')
    with engine.connect() as conn, conn.begin():
        conn.execute(images.insert().values(shared_temp))
    with engine.connect() as conn:
        rows = conn.execute(images.select().where(images.c.id.like('%_new_to_old')).order_by(images.c.id)).fetchall()
    self.assertEqual(3, len(rows))
    self.assertEqual(0, rows[0].is_public)
    self.assertEqual('private_id_new_to_old', rows[0].id)
    self.assertEqual('private', rows[0].visibility)
    self.assertEqual(1, rows[1].is_public)
    self.assertEqual('public_id_new_to_old', rows[1].id)
    self.assertEqual('public', rows[1].visibility)
    self.assertEqual(0, rows[2].is_public)
    self.assertEqual('shared_id_new_to_old', rows[2].id)
    self.assertEqual('shared', rows[2].visibility)