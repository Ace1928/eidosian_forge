from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
class TestMigrationMultipleExtensions(test_base.BaseTestCase):

    def setUp(self):
        self.migration_config = {'alembic_ini_path': '.', 'migrate_repo_path': '.', 'db_url': 'sqlite://'}
        engine = sqlalchemy.create_engine(self.migration_config['db_url'])
        self.migration_manager = manager.MigrationManager(self.migration_config, engine)
        self.first_ext = MockWithCmp()
        self.first_ext.obj.order = 1
        self.first_ext.obj.upgrade.return_value = 100
        self.first_ext.obj.downgrade.return_value = 0
        self.second_ext = MockWithCmp()
        self.second_ext.obj.order = 2
        self.second_ext.obj.upgrade.return_value = 200
        self.second_ext.obj.downgrade.return_value = 100
        self.migration_manager._manager.extensions = [self.first_ext, self.second_ext]
        super(TestMigrationMultipleExtensions, self).setUp()

    def test_upgrade_right_order(self):
        results = self.migration_manager.upgrade(None)
        self.assertEqual([100, 200], results)

    def test_downgrade_right_order(self):
        results = self.migration_manager.downgrade(None)
        self.assertEqual([100, 0], results)

    def test_upgrade_does_not_go_too_far(self):
        self.first_ext.obj.has_revision.return_value = True
        self.second_ext.obj.has_revision.return_value = False
        self.second_ext.obj.upgrade.side_effect = AssertionError('this method should not have been called')
        results = self.migration_manager.upgrade(100)
        self.assertEqual([100], results)

    def test_downgrade_does_not_go_too_far(self):
        self.second_ext.obj.has_revision.return_value = True
        self.first_ext.obj.has_revision.return_value = False
        self.first_ext.obj.downgrade.side_effect = AssertionError('this method should not have been called')
        results = self.migration_manager.downgrade(100)
        self.assertEqual([100], results)

    def test_upgrade_checks_rev_existence(self):
        self.first_ext.obj.has_revision.return_value = False
        self.second_ext.obj.has_revision.return_value = False
        self.assertRaises(exception.DBMigrationError, self.migration_manager.upgrade, 100)
        self.assertEqual([100, 200], self.migration_manager.upgrade(None))
        self.second_ext.obj.has_revision.return_value = True
        self.assertEqual([100, 200], self.migration_manager.upgrade(200))
        self.assertEqual([100, 200], self.migration_manager.upgrade(None))

    def test_downgrade_checks_rev_existence(self):
        self.first_ext.obj.has_revision.return_value = False
        self.second_ext.obj.has_revision.return_value = False
        self.assertRaises(exception.DBMigrationError, self.migration_manager.downgrade, 100)
        self.assertEqual([100, 0], self.migration_manager.downgrade(None))
        self.first_ext.obj.has_revision.return_value = True
        self.assertEqual([100, 0], self.migration_manager.downgrade(200))
        self.assertEqual([100, 0], self.migration_manager.downgrade(None))
        self.assertEqual([100, 0], self.migration_manager.downgrade('base'))