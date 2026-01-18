import datetime
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _check_mitaka02(self, engine, data):
    metadef_resource_types = db_utils.get_table(engine, 'metadef_resource_types')
    with engine.connect() as conn:
        result = conn.execute(metadef_resource_types.select().where(metadef_resource_types.c.name == 'OS::Nova::Instance')).fetchall()
        self.assertEqual(0, len(result))
        result = conn.execute(metadef_resource_types.select().where(metadef_resource_types.c.name == 'OS::Nova::Server')).fetchall()
        self.assertEqual(1, len(result))