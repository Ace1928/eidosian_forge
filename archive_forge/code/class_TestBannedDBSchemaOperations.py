from alembic import command as alembic_api
from alembic import script as alembic_script
import fixtures
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from oslotest import base as test_base
import sqlalchemy
import testtools
from heat.db import migration
from heat.db import models
class TestBannedDBSchemaOperations(testtools.TestCase):

    def test_column(self):
        column = sqlalchemy.Column()
        with BannedDBSchemaOperations(['Column']):
            self.assertRaises(DBNotAllowed, column.drop)
            self.assertRaises(DBNotAllowed, column.alter)

    def test_table(self):
        table = sqlalchemy.Table('foo', sqlalchemy.MetaData())
        with BannedDBSchemaOperations(['Table']):
            self.assertRaises(DBNotAllowed, table.drop)
            self.assertRaises(DBNotAllowed, table.alter)