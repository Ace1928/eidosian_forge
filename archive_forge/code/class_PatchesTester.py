import os
from breezy.errors import BinaryFile
from breezy.iterablefile import IterableFile
from breezy.patch import (PatchInvokeError, diff3, iter_patched_from_hunks,
from breezy.patches import parse_patch
from breezy.tests import TestCase, TestCaseInTempDir
class PatchesTester(TestCase):

    def datafile(self, filename):
        data_path = os.path.join(os.path.dirname(__file__), 'test_patches_data', filename)
        return open(data_path, 'rb')

    def data_lines(self, filename):
        with self.datafile(filename) as datafile:
            return datafile.readlines()

    def test_iter_patched_from_hunks(self):
        """Test a few patch files, and make sure they work."""
        files = [('diff-2', 'orig-2', 'mod-2'), ('diff-3', 'orig-3', 'mod-3'), ('diff-4', 'orig-4', 'mod-4'), ('diff-5', 'orig-5', 'mod-5'), ('diff-6', 'orig-6', 'mod-6'), ('diff-7', 'orig-7', 'mod-7')]
        for diff, orig, mod in files:
            parsed = parse_patch(self.datafile(diff))
            orig_lines = list(self.datafile(orig))
            mod_lines = list(self.datafile(mod))
            iter_patched = iter_patched_from_hunks(orig_lines, parsed.hunks)
            patched_file = IterableFile(iter_patched)
            count = 0
            for patch_line in patched_file:
                self.assertEqual(patch_line, mod_lines[count], 'for file %s' % diff)
                count += 1
            self.assertEqual(count, len(mod_lines))