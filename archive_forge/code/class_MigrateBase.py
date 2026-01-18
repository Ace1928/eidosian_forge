import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslo_log import fixture as log_fixture
from oslo_log import log
import sqlalchemy.exc
from keystone.cmd import cli
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
class MigrateBase(db_fixtures.OpportunisticDBTestMixin):
    """Test complete orchestration between all database phases."""

    def setUp(self):
        super().setUp()
        self.useFixture(log_fixture.get_logging_handle_error_fixture())
        self.stdlog = self.useFixture(ksfixtures.StandardLogging())
        self.useFixture(ksfixtures.WarningsFixture())
        self.engine = enginefacade.writer.get_engine()
        self.sessionmaker = enginefacade.writer.get_sessionmaker()
        db_options.set_defaults(CONF, connection=self.engine.url)
        sql.core._TESTING_USE_GLOBAL_CONTEXT_MANAGER = True
        self.addCleanup(setattr, sql.core, '_TESTING_USE_GLOBAL_CONTEXT_MANAGER', False)
        self.addCleanup(sql.cleanup)

    def expand(self):
        """Expand database schema."""
        upgrades.expand_schema(engine=self.engine)

    def contract(self):
        """Contract database schema."""
        upgrades.contract_schema(engine=self.engine)

    @property
    def metadata(self):
        """A collection of tables and their associated schemas."""
        return sqlalchemy.MetaData()

    def load_table(self, name):
        table = sqlalchemy.Table(name, self.metadata, autoload_with=self.engine)
        return table

    def assertTableDoesNotExist(self, table_name):
        """Assert that a given table exists cannot be selected by name."""
        try:
            sqlalchemy.Table(table_name, self.metadata, autoload_with=self.engine)
        except sqlalchemy.exc.NoSuchTableError:
            pass
        else:
            raise AssertionError('Table "%s" already exists' % table_name)

    def assertTableColumns(self, table_name, expected_cols):
        """Assert that the table contains the expected set of columns."""
        table = self.load_table(table_name)
        actual_cols = [col.name for col in table.columns]
        self.assertCountEqual(expected_cols, actual_cols, '%s table' % table_name)

    def test_db_sync_check(self):
        checker = cli.DbSync()
        log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
        status = checker.check_db_sync_status()
        self.assertIn('keystone-manage db_sync --expand', log_info.output)
        self.assertEqual(status, 2)
        self.expand()
        log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
        status = checker.check_db_sync_status()
        self.assertIn('keystone-manage db_sync --contract', log_info.output)
        self.assertEqual(status, 4)
        self.contract()
        log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
        status = checker.check_db_sync_status()
        self.assertIn('All db_sync commands are upgraded', log_info.output)
        self.assertEqual(status, 0)

    def test_upgrade_add_initial_tables(self):
        self.expand()
        for table in INITIAL_TABLE_STRUCTURE:
            self.assertTableColumns(table, INITIAL_TABLE_STRUCTURE[table])