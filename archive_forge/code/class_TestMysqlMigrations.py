import collections
import os
from alembic import command as alembic_command
from alembic import script as alembic_script
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from sqlalchemy import sql
import sqlalchemy.types as types
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import versions
from glance.db.sqlalchemy import models
from glance.db.sqlalchemy import models_metadef
import glance.tests.utils as test_utils
class TestMysqlMigrations(test_fixtures.OpportunisticDBTestMixin, AlembicMigrationsMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.MySQLOpportunisticFixture

    def test_mysql_innodb_tables(self):
        test_utils.db_sync(engine=self.engine)
        with self.engine.connect() as conn:
            total = conn.execute(sql.text('SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=:database'), {'database': self.engine.url.database})
        self.assertGreater(total.scalar(), 0, 'No tables found. Wrong schema?')
        with self.engine.connect() as conn:
            noninnodb = conn.execute(sql.text("SELECT count(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=:database AND ENGINE!='InnoDB' AND TABLE_NAME!='migrate_version'"), {'database': self.engine.url.database})
            count = noninnodb.scalar()
        self.assertEqual(0, count, '%d non InnoDB tables created' % count)