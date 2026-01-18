from io import BytesIO
from dulwich.tests import TestCase
from ..objects import ZERO_SHA
from ..reflog import (
class ReflogDropTests(TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.f = BytesIO(_TEST_REFLOG)
        self.original_log = list(read_reflog(self.f))
        self.f.seek(0)

    def _read_log(self):
        self.f.seek(0)
        return list(read_reflog(self.f))

    def test_invalid(self):
        self.assertRaises(ValueError, drop_reflog_entry, self.f, -1)

    def test_drop_entry(self):
        drop_reflog_entry(self.f, 0)
        log = self._read_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(self.original_log[0:2], log)
        self.f.seek(0)
        drop_reflog_entry(self.f, 1)
        log = self._read_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(self.original_log[1], log[0])

    def test_drop_entry_with_rewrite(self):
        drop_reflog_entry(self.f, 1, True)
        log = self._read_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(self.original_log[0], log[0])
        self.assertEqual(self.original_log[0].new_sha, log[1].old_sha)
        self.assertEqual(self.original_log[2].new_sha, log[1].new_sha)
        self.f.seek(0)
        drop_reflog_entry(self.f, 1, True)
        log = self._read_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(ZERO_SHA, log[0].old_sha)
        self.assertEqual(self.original_log[2].new_sha, log[0].new_sha)