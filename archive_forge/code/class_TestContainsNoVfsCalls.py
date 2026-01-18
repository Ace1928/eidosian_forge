from testtools.matchers import *
from ...tests import CapturedCall, TestCase
from ..smart.client import CallHookParams
from .matchers import *
class TestContainsNoVfsCalls(TestCase):

    def _make_call(self, method, args):
        return CapturedCall(CallHookParams(method, args, None, None, None), 0)

    def test__str__(self):
        self.assertEqual('ContainsNoVfsCalls()', str(ContainsNoVfsCalls()))

    def test_empty(self):
        self.assertIs(None, ContainsNoVfsCalls().match([]))

    def test_no_vfs_calls(self):
        calls = [self._make_call('Branch.get_config_file', [])]
        self.assertIs(None, ContainsNoVfsCalls().match(calls))

    def test_ignores_unknown(self):
        calls = [self._make_call('unknown', [])]
        self.assertIs(None, ContainsNoVfsCalls().match(calls))

    def test_match(self):
        calls = [self._make_call(b'append', [b'file']), self._make_call(b'Branch.get_config_file', [])]
        mismatch = ContainsNoVfsCalls().match(calls)
        self.assertIsNot(None, mismatch)
        self.assertEqual([calls[0].call], mismatch.vfs_calls)
        self.assertIn(mismatch.describe(), ["no VFS calls expected, got: b'append'(b'file')", "no VFS calls expected, got: append('file')"])