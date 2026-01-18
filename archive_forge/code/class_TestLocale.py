import sys
import unittest
from breezy import tests
class TestLocale(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        if sys.platform in ('win32',):
            raise tests.TestSkipped('Windows does not respond to the LANG env variable')
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/a'])
        tree.add('a')
        tree.commit('Unicode µ commit', rev_id=b'r1', committer='جوجو Meinel <juju@info.com>', timestamp=1156451297.96, timezone=0)
        self.tree = tree

    def run_log_quiet_long(self, args, env_changes={}):
        cmd = ['--no-aliases', '--no-plugins', '-Oprogress_bar=none', 'log', '-q', '--log-format=long']
        cmd.extend(args)
        return self.run_brz_subprocess(cmd, env_changes=env_changes)

    @unittest.skip('encoding when LANG=C is currently borked')
    def test_log_coerced_utf8(self):
        self.disable_missing_extensions_warning()
        out, err = self.run_log_quiet_long(['tree'], env_changes={'LANG': 'C', 'LC_ALL': 'C', 'LC_CTYPE': None, 'LANGUAGE': None})
        self.assertEqual(b'', err)
        self.assertEqualDiff(b'------------------------------------------------------------\nrevno: 1\ncommitter: \xd8\xac\xd9\x88\xd8\xac\xd9\x88 Meinel <juju@info.com>\nbranch nick: tree\ntimestamp: Thu 2006-08-24 20:28:17 +0000\nmessage:\n  Unicode \xc2\xb5 commit\n', out)

    @unittest.skipIf(sys.version_info[:2] >= (3, 8), "python > 3.8 doesn't allow changing filesystem default encoding")
    def test_log_C(self):
        self.disable_missing_extensions_warning()
        out, err = self.run_log_quiet_long(['tree'], env_changes={'LANG': 'C', 'LC_ALL': 'C', 'LC_CTYPE': None, 'LANGUAGE': None, 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '0'})
        self.assertEqual(b'', err)
        self.assertEqualDiff(b'------------------------------------------------------------\nrevno: 1\ncommitter: ???? Meinel <juju@info.com>\nbranch nick: tree\ntimestamp: Thu 2006-08-24 20:28:17 +0000\nmessage:\n  Unicode ? commit\n', out)

    @unittest.skipIf(sys.version_info[:2] >= (3, 8), "python > 3.8 doesn't allow changing filesystem default encoding")
    def test_log_BOGUS(self):
        out, err = self.run_log_quiet_long(['tree'], env_changes={'LANG': 'BOGUS', 'LC_ALL': None, 'LC_CTYPE': None, 'LANGUAGE': None, 'PYTHONCOERCECLOCALE': '0', 'PYTHONUTF8': '0'})
        self.assertStartsWith(err, b'brz: WARNING: Error: unsupported locale setting')
        self.assertEqualDiff(b'------------------------------------------------------------\nrevno: 1\ncommitter: ???? Meinel <juju@info.com>\nbranch nick: tree\ntimestamp: Thu 2006-08-24 20:28:17 +0000\nmessage:\n  Unicode ? commit\n', out)