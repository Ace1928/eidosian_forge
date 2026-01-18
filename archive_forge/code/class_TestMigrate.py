from contextlib import contextmanager
import os
import sqlite3
import tempfile
import time
from unittest import mock
import uuid
from oslo_config import cfg
from glance import sqlite_migration
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestMigrate(test_utils.BaseTestCase):

    def _store_dir(self, store):
        return os.path.join(self.test_dir, store)

    def setUp(self):
        super(TestMigrate, self).setUp()
        self.config(worker_self_reference_url='http://worker1.example.com')
        fd, self.db = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        self.db_api = unit_test_utils.FakeDB(initialize=False)
        self.migrate = sqlite_migration.Migrate(self.db, self.db_api)
        self.addCleanup(self.drop_db)

    def drop_db(self):
        if os.path.exists(self.db):
            os.remove(self.db)

    def create_db(self):
        conn = sqlite3.connect(self.db, check_same_thread=False, factory=sqlite3.Connection)
        conn.executescript('\n            CREATE TABLE IF NOT EXISTS cached_images (\n                image_id TEXT PRIMARY KEY,\n                last_accessed REAL DEFAULT 0.0,\n                last_modified REAL DEFAULT 0.0,\n                size INTEGER DEFAULT 0,\n                hits INTEGER DEFAULT 0,\n                checksum TEXT\n            );\n        ')
        conn.close()

    @contextmanager
    def get_db(self):
        """
        Returns a context manager that produces a database connection that
        self-closes and calls rollback if an error occurs while using the
        database connection
        """
        conn = sqlite3.connect(self.db, check_same_thread=False, factory=sqlite3.Connection)
        conn.row_factory = sqlite3.Row
        conn.text_factory = str
        conn.execute('PRAGMA synchronous = NORMAL')
        conn.execute('PRAGMA count_changes = OFF')
        conn.execute('PRAGMA temp_store = MEMORY')
        try:
            yield conn
        except sqlite3.DatabaseError:
            conn.rollback()
        finally:
            conn.close()

    def initialize_fake_cache_details(self):
        with self.get_db() as sq_db:
            filesize = 100
            now = time.time()
            sq_db.execute('INSERT INTO cached_images (image_id,\n                last_accessed, last_modified, hits, size)\n                VALUES (?, ?, ?, ?, ?)', (FAKE_IMAGE_1, now, now, 0, filesize))
            sq_db.commit()

    def test_migrate_if_required_false(self):
        self.config(image_cache_driver='sqlite')
        self.assertFalse(sqlite_migration.migrate_if_required())

    def test_migrate_if_required_cache_disabled(self):
        self.config(flavor='keystone', group='paste_deploy')
        self.config(image_cache_driver='centralized_db')
        self.assertFalse(sqlite_migration.migrate_if_required())

    @mock.patch('os.path.exists')
    @mock.patch('os.path.join', new=mock.MagicMock())
    def test_migrate_if_required_db_not_found(self, mock_exists):
        mock_exists.return_value = False
        self.config(flavor='keystone+cache', group='paste_deploy')
        self.config(image_cache_driver='centralized_db')
        with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
            sqlite_migration.migrate_if_required()
            mock_log.debug.assert_called_once_with('SQLite caching database not located, skipping migration')

    def test_migrate_empty_db(self):
        with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
            self.migrate.migrate()
            expected_calls = [mock.call('Adding local node reference %(node)s in centralized db', {'node': 'http://worker1.example.com'}), mock.call('Connecting to SQLite db %s', self.db)]
            mock_log.debug.assert_has_calls(expected_calls)

    def test_migrate_duplicate_node_reference(self):
        self.migrate.migrate()
        with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
            self.migrate.migrate()
            expected_calls = [mock.call('Adding local node reference %(node)s in centralized db', {'node': 'http://worker1.example.com'}), mock.call('Node reference %(node)s is already recorded, ignoring it', {'node': 'http://worker1.example.com'}), mock.call('Connecting to SQLite db %s', self.db)]
            mock_log.debug.assert_has_calls(expected_calls)

    def test_migrate_record_exists_in_centralized_db(self):
        self.create_db()
        self.initialize_fake_cache_details()
        with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
            with mock.patch.object(self.db_api, 'is_image_cached_for_node') as mock_call:
                mock_call.return_value = True
                self.migrate.migrate()
            expected_calls = [mock.call('Adding local node reference %(node)s in centralized db', {'node': 'http://worker1.example.com'}), mock.call('Connecting to SQLite db %s', self.db), mock.call('Skipping migrating image %(uuid)s from SQLite to Centralized db for node %(node)s as it is present in Centralized db.', {'uuid': FAKE_IMAGE_1, 'node': 'http://worker1.example.com'})]
            mock_log.debug.assert_has_calls(expected_calls)

    def test_migrate(self):
        self.config(image_cache_driver='centralized_db')
        self.create_db()
        self.initialize_fake_cache_details()
        with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
            self.migrate.migrate()
            expected_calls = [mock.call('Adding local node reference %(node)s in centralized db', {'node': 'http://worker1.example.com'}), mock.call('Connecting to SQLite db %s', self.db), mock.call('Migrating image %s from SQLite to Centralized db.', FAKE_IMAGE_1), mock.call('Image %(uuid)s is migrated to centralized db for node %(node)s', {'uuid': FAKE_IMAGE_1, 'node': 'http://worker1.example.com'}), mock.call('Deleting image %s from SQLite db', FAKE_IMAGE_1), mock.call('Migrated %d records from SQLite db to Centralized db', 1)]
            mock_log.debug.assert_has_calls(expected_calls)