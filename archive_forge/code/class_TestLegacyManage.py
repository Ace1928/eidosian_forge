import io
from unittest import mock
import fixtures
from glance.cmd import manage
from glance.common import exception
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata as db_metadata
from glance.tests import utils as test_utils
from sqlalchemy.engine.url import make_url as sqlalchemy_make_url
class TestLegacyManage(TestManageBase):

    @mock.patch.object(manage.DbCommands, 'version')
    def test_legacy_db_version(self, db_upgrade):
        self._main_test_helper(['glance.cmd.manage', 'db_version'], manage.DbCommands.version)

    @mock.patch.object(manage.DbCommands, 'sync')
    def test_legacy_db_sync(self, db_sync):
        self._main_test_helper(['glance.cmd.manage', 'db_sync'], manage.DbCommands.sync, None)

    @mock.patch.object(manage.DbCommands, 'upgrade')
    def test_legacy_db_upgrade(self, db_upgrade):
        self._main_test_helper(['glance.cmd.manage', 'db_upgrade'], manage.DbCommands.upgrade, None)

    @mock.patch.object(manage.DbCommands, 'version_control')
    def test_legacy_db_version_control(self, db_version_control):
        self._main_test_helper(['glance.cmd.manage', 'db_version_control'], manage.DbCommands.version_control, None)

    @mock.patch.object(manage.DbCommands, 'sync')
    def test_legacy_db_sync_version(self, db_sync):
        self._main_test_helper(['glance.cmd.manage', 'db_sync', 'liberty'], manage.DbCommands.sync, 'liberty')

    @mock.patch.object(manage.DbCommands, 'upgrade')
    def test_legacy_db_upgrade_version(self, db_upgrade):
        self._main_test_helper(['glance.cmd.manage', 'db_upgrade', 'liberty'], manage.DbCommands.upgrade, 'liberty')

    @mock.patch.object(manage.DbCommands, 'expand')
    def test_legacy_db_expand(self, db_expand):
        self._main_test_helper(['glance.cmd.manage', 'db_expand'], manage.DbCommands.expand)

    @mock.patch.object(manage.DbCommands, 'migrate')
    def test_legacy_db_migrate(self, db_migrate):
        self._main_test_helper(['glance.cmd.manage', 'db_migrate'], manage.DbCommands.migrate)

    @mock.patch.object(manage.DbCommands, 'contract')
    def test_legacy_db_contract(self, db_contract):
        self._main_test_helper(['glance.cmd.manage', 'db_contract'], manage.DbCommands.contract)

    def test_db_metadefs_unload(self):
        db_metadata.db_unload_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_unload_metadefs'], db_metadata.db_unload_metadefs, db_api.get_engine())

    def test_db_metadefs_load(self):
        db_metadata.db_load_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_load_metadefs'], db_metadata.db_load_metadefs, db_api.get_engine(), None, None, None, None)

    def test_db_metadefs_load_with_specified_path(self):
        db_metadata.db_load_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_load_metadefs', '/mock/'], db_metadata.db_load_metadefs, db_api.get_engine(), '/mock/', None, None, None)

    def test_db_metadefs_load_from_path_merge(self):
        db_metadata.db_load_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_load_metadefs', '/mock/', 'True'], db_metadata.db_load_metadefs, db_api.get_engine(), '/mock/', 'True', None, None)

    def test_db_metadefs_load_from_merge_and_prefer_new(self):
        db_metadata.db_load_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_load_metadefs', '/mock/', 'True', 'True'], db_metadata.db_load_metadefs, db_api.get_engine(), '/mock/', 'True', 'True', None)

    def test_db_metadefs_load_from_merge_and_prefer_new_and_overwrite(self):
        db_metadata.db_load_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_load_metadefs', '/mock/', 'True', 'True', 'True'], db_metadata.db_load_metadefs, db_api.get_engine(), '/mock/', 'True', 'True', 'True')

    def test_db_metadefs_export(self):
        db_metadata.db_export_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_export_metadefs'], db_metadata.db_export_metadefs, db_api.get_engine(), None)

    def test_db_metadefs_export_with_specified_path(self):
        db_metadata.db_export_metadefs = mock.Mock()
        self._main_test_helper(['glance.cmd.manage', 'db_export_metadefs', '/mock/'], db_metadata.db_export_metadefs, db_api.get_engine(), '/mock/')