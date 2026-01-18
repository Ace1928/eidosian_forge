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
class TestMigrations(test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):

    def test_no_downgrade(self):
        migrate_file = versions.__path__[0]
        for parent, dirnames, filenames in os.walk(migrate_file):
            for filename in filenames:
                if filename.split('.')[1] == 'py':
                    model_name = filename.split('.')[0]
                    model = __import__('glance.db.sqlalchemy.alembic_migrations.versions.' + model_name)
                    obj = getattr(getattr(getattr(getattr(getattr(model, 'db'), 'sqlalchemy'), 'alembic_migrations'), 'versions'), model_name)
                    func = getattr(obj, 'downgrade', None)
                    self.assertIsNone(func)