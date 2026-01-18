from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
class TestContent(TestCase):
    """Test serialization of the content part of a line."""

    def test_dir(self):
        entry = inventory.make_entry('directory', 'a dir', None)
        self.assertEqual(b'dir', inventory_delta._directory_content(entry))

    def test_file_0_short_sha(self):
        file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
        file_entry.text_sha1 = b''
        file_entry.text_size = 0
        self.assertEqual(b'file\x000\x00\x00', inventory_delta._file_content(file_entry))

    def test_file_10_foo(self):
        file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
        file_entry.text_sha1 = b'foo'
        file_entry.text_size = 10
        self.assertEqual(b'file\x0010\x00\x00foo', inventory_delta._file_content(file_entry))

    def test_file_executable(self):
        file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
        file_entry.executable = True
        file_entry.text_sha1 = b'foo'
        file_entry.text_size = 10
        self.assertEqual(b'file\x0010\x00Y\x00foo', inventory_delta._file_content(file_entry))

    def test_file_without_size(self):
        file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
        file_entry.text_sha1 = b'foo'
        self.assertRaises(InventoryDeltaError, inventory_delta._file_content, file_entry)

    def test_file_without_sha1(self):
        file_entry = inventory.make_entry('file', 'a file', None, b'file-id')
        file_entry.text_size = 10
        self.assertRaises(InventoryDeltaError, inventory_delta._file_content, file_entry)

    def test_link_empty_target(self):
        entry = inventory.make_entry('symlink', 'a link', None)
        entry.symlink_target = ''
        self.assertEqual(b'link\x00', inventory_delta._link_content(entry))

    def test_link_unicode_target(self):
        entry = inventory.make_entry('symlink', 'a link', None)
        entry.symlink_target = b' \xc3\xa5'.decode('utf8')
        self.assertEqual(b'link\x00 \xc3\xa5', inventory_delta._link_content(entry))

    def test_link_space_target(self):
        entry = inventory.make_entry('symlink', 'a link', None)
        entry.symlink_target = ' '
        self.assertEqual(b'link\x00 ', inventory_delta._link_content(entry))

    def test_link_no_target(self):
        entry = inventory.make_entry('symlink', 'a link', None)
        self.assertRaises(InventoryDeltaError, inventory_delta._link_content, entry)

    def test_reference_null(self):
        entry = inventory.make_entry('tree-reference', 'a tree', None)
        entry.reference_revision = NULL_REVISION
        self.assertEqual(b'tree\x00null:', inventory_delta._reference_content(entry))

    def test_reference_revision(self):
        entry = inventory.make_entry('tree-reference', 'a tree', None)
        entry.reference_revision = b'foo@\xc3\xa5b-lah'
        self.assertEqual(b'tree\x00foo@\xc3\xa5b-lah', inventory_delta._reference_content(entry))

    def test_reference_no_reference(self):
        entry = inventory.make_entry('tree-reference', 'a tree', None)
        self.assertRaises(InventoryDeltaError, inventory_delta._reference_content, entry)