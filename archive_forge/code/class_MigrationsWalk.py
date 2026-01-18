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
class MigrationsWalk(test_fixtures.OpportunisticDBTestMixin, test_base.BaseTestCase):
    TIMEOUT_SCALING_FACTOR = 4

    def setUp(self):
        super().setUp()
        self.engine = enginefacade.writer.get_engine()
        self.config = migration._find_alembic_conf()
        self.init_version = migration.ALEMBIC_INIT_VERSION

    def _migrate_up(self, revision, connection):
        check_method = getattr(self, f'_check_{revision}', None)
        if revision != self.init_version:
            self.assertIsNotNone(check_method, f"DB Migration {revision} doesn't have a test; add one")
        pre_upgrade = getattr(self, f'_pre_upgrade_{revision}', None)
        if pre_upgrade:
            pre_upgrade(connection)
        alembic_api.upgrade(self.config, revision)
        if check_method:
            check_method(connection)

    def test_walk_versions(self):
        with self.engine.begin() as connection:
            self.config.attributes['connection'] = connection
            script = alembic_script.ScriptDirectory.from_config(self.config)
            revisions = list(script.walk_revisions())
            revisions.reverse()
            for revision_script in revisions:
                self._migrate_up(revision_script.revision, connection)