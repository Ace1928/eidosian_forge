import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestPackStat(tests.TestCase):
    """Check packed representaton of stat values is robust on all inputs"""
    scenarios = helper_scenarios

    def pack(self, statlike_tuple):
        return self.helpers.pack_stat(os.stat_result(statlike_tuple))

    @staticmethod
    def unpack_field(packed_string, stat_field):
        return _dirstate_helpers_py._unpack_stat(packed_string)[stat_field]

    def test_result(self):
        self.assertEqual(b'AAAQAAAAABAAAAARAAAAAgAAAAEAAIHk', self.pack((33252, 1, 2, 0, 0, 0, 4096, 15.5, 16.5, 17.5)))

    def test_giant_inode(self):
        packed = self.pack((33252, 66571995836, 0, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(2147486396, self.unpack_field(packed, 'st_ino'))

    def test_giant_size(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, (1 << 33) + 4096, 0, 0, 0))
        self.assertEqual(4096, self.unpack_field(packed, 'st_size'))

    def test_fractional_mtime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 16.9375, 0))
        self.assertEqual(16, self.unpack_field(packed, 'st_mtime'))

    def test_ancient_mtime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, -11644473600.0, 0))
        self.assertEqual(1240428288, self.unpack_field(packed, 'st_mtime'))

    def test_distant_mtime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 64060588800.0, 0))
        self.assertEqual(3931046656, self.unpack_field(packed, 'st_mtime'))

    def test_fractional_ctime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 0, 17.5625))
        self.assertEqual(17, self.unpack_field(packed, 'st_ctime'))

    def test_ancient_ctime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 0, -11644473600.0))
        self.assertEqual(1240428288, self.unpack_field(packed, 'st_ctime'))

    def test_distant_ctime(self):
        packed = self.pack((33252, 0, 0, 0, 0, 0, 0, 0, 0, 64060588800.0))
        self.assertEqual(3931046656, self.unpack_field(packed, 'st_ctime'))

    def test_negative_dev(self):
        packed = self.pack((33252, 0, -4294966494, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(802, self.unpack_field(packed, 'st_dev'))