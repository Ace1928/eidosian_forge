from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import utils as db_utils
import sqlalchemy
from glance.tests.functional.db import test_migrations
import glance.tests.utils as test_utils
def _pre_upgrade_pike_contract01(self, engine):
    for table_name in self.artifacts_table_names:
        table = db_utils.get_table(engine, table_name)
        self.assertIsNotNone(table)