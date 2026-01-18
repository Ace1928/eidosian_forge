from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
class TestDescribeChanges(TestCase):

    def test_describe_change(self):
        old_a = InventoryFile(b'a-id', 'a_file', ROOT_ID)
        old_a.text_sha1 = b'123132'
        old_a.text_size = 0
        new_a = InventoryFile(b'a-id', 'a_file', ROOT_ID)
        new_a.text_sha1 = b'123132'
        new_a.text_size = 0
        self.assertChangeDescription('unchanged', old_a, new_a)
        new_a.text_size = 10
        new_a.text_sha1 = b'abcabc'
        self.assertChangeDescription('modified', old_a, new_a)
        self.assertChangeDescription('added', None, new_a)
        self.assertChangeDescription('removed', old_a, None)
        self.assertChangeDescription('unchanged', None, None)
        new_a.name = 'newfilename'
        self.assertChangeDescription('modified and renamed', old_a, new_a)
        new_a.name = old_a.name
        new_a.parent_id = b'somedir-id'
        self.assertChangeDescription('modified and renamed', old_a, new_a)
        new_a.text_size = old_a.text_size
        new_a.text_sha1 = old_a.text_sha1
        new_a.name = old_a.name
        new_a.name = 'newfilename'
        self.assertChangeDescription('renamed', old_a, new_a)
        new_a.name = old_a.name
        new_a.parent_id = b'somedir-id'
        self.assertChangeDescription('renamed', old_a, new_a)

    def assertChangeDescription(self, expected_change, old_ie, new_ie):
        change = InventoryEntry.describe_change(old_ie, new_ie)
        self.assertEqual(expected_change, change)