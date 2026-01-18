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
class ModelsMigrationsSyncPostgres(ModelsMigrationSyncMixin, test_migrations.ModelsMigrationsSync, test_fixtures.OpportunisticDBTestMixin, test_utils.BaseTestCase):
    FIXTURE = test_fixtures.PostgresqlOpportunisticFixture