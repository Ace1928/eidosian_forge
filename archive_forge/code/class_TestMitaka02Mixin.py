import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
class TestMitaka02Mixin(test_migrations.AlembicMigrationsMixin):

    def _pre_upgrade_mitaka02(self, engine):
        metadef_resource_types = db_utils.get_table(engine, 'metadef_resource_types')
        now = datetime.datetime.now()
        db_rec1 = dict(id='9580', name='OS::Nova::Instance', protected=False, created_at=now, updated_at=now)
        db_rec2 = dict(id='9581', name='OS::Nova::Blah', protected=False, created_at=now, updated_at=now)
        db_values = (db_rec1, db_rec2)
        with engine.connect() as conn, conn.begin():
            conn.execute(metadef_resource_types.insert().values(db_values))

    def _check_mitaka02(self, engine, data):
        metadef_resource_types = db_utils.get_table(engine, 'metadef_resource_types')
        with engine.connect() as conn:
            result = conn.execute(metadef_resource_types.select().where(metadef_resource_types.c.name == 'OS::Nova::Instance')).fetchall()
            self.assertEqual(0, len(result))
            result = conn.execute(metadef_resource_types.select().where(metadef_resource_types.c.name == 'OS::Nova::Server')).fetchall()
            self.assertEqual(1, len(result))