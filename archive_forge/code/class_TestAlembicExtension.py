from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
@mock.patch('oslo_db.sqlalchemy.migration_cli.ext_alembic.alembic.command')
class TestAlembicExtension(test_base.BaseTestCase):

    def setUp(self):
        self.migration_config = {'alembic_ini_path': '.', 'db_url': 'sqlite://'}
        self.engine = sqlalchemy.create_engine(self.migration_config['db_url'])
        self.alembic = ext_alembic.AlembicExtension(self.engine, self.migration_config)
        super(TestAlembicExtension, self).setUp()

    def test_check_enabled_true(self, command):
        """Check enabled returns True

        Verifies that enabled returns True on non empty
        alembic_ini_path conf variable
        """
        self.assertTrue(self.alembic.enabled)

    def test_check_enabled_false(self, command):
        """Check enabled returns False

        Verifies enabled returns False on empty alembic_ini_path variable
        """
        self.migration_config['alembic_ini_path'] = ''
        alembic = ext_alembic.AlembicExtension(self.engine, self.migration_config)
        self.assertFalse(alembic.enabled)

    def test_upgrade_none(self, command):
        self.alembic.upgrade(None)
        command.upgrade.assert_called_once_with(self.alembic.config, 'head')

    def test_upgrade_normal(self, command):
        self.alembic.upgrade('131daa')
        command.upgrade.assert_called_once_with(self.alembic.config, '131daa')

    def test_downgrade_none(self, command):
        self.alembic.downgrade(None)
        command.downgrade.assert_called_once_with(self.alembic.config, 'base')

    def test_downgrade_int(self, command):
        self.alembic.downgrade(111)
        command.downgrade.assert_called_once_with(self.alembic.config, 'base')

    def test_downgrade_normal(self, command):
        self.alembic.downgrade('131daa')
        command.downgrade.assert_called_once_with(self.alembic.config, '131daa')

    def test_revision(self, command):
        self.alembic.revision(message='test', autogenerate=True)
        command.revision.assert_called_once_with(self.alembic.config, message='test', autogenerate=True)

    def test_stamp(self, command):
        self.alembic.stamp('stamp')
        command.stamp.assert_called_once_with(self.alembic.config, revision='stamp')

    def test_version(self, command):
        version = self.alembic.version()
        self.assertIsNone(version)

    def test_has_revision(self, command):
        with mock.patch('oslo_db.sqlalchemy.migration_cli.ext_alembic.alembic_script') as mocked:
            self.alembic.config.get_main_option = mock.Mock()
            self.assertIs(True, self.alembic.has_revision('test'))
            self.alembic.config.get_main_option.assert_called_once_with('script_location')
            mocked.ScriptDirectory().get_revision.assert_called_once_with('test')
            self.assertIs(True, self.alembic.has_revision(None))
            self.assertIs(True, self.alembic.has_revision('head'))
            self.assertIs(True, self.alembic.has_revision('+1'))

    def test_has_revision_negative(self, command):
        with mock.patch('oslo_db.sqlalchemy.migration_cli.ext_alembic.alembic_script') as mocked:
            mocked.ScriptDirectory().get_revision.side_effect = alembic.util.CommandError
            self.alembic.config.get_main_option = mock.Mock()
            self.assertIs(False, self.alembic.has_revision('test'))
            self.alembic.config.get_main_option.assert_called_once_with('script_location')
            mocked.ScriptDirectory().get_revision.assert_called_once_with('test')