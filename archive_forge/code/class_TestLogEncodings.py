import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogEncodings(tests.TestCaseInTempDir):
    _mu = 'µ'
    _message = 'Message with µ'
    good_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp437', 'cp1251', 'cp1258']
    bad_encodings = ['ascii', 'iso-8859-2', 'koi8_r']

    def setUp(self):
        super().setUp()
        self.overrideAttr(osutils, '_cached_user_encoding')

    def create_branch(self):
        brz = self.run_bzr
        brz('init')
        self.build_tree_contents([('a', b'some stuff\n')])
        brz('add a')
        brz(['commit', '-m', self._message])

    def try_encoding(self, encoding, fail=False):
        brz = self.run_bzr
        if fail:
            self.assertRaises(UnicodeEncodeError, self._mu.encode, encoding)
            encoded_msg = self._message.encode(encoding, 'replace')
        else:
            encoded_msg = self._message.encode(encoding)
        old_encoding = osutils._cached_user_encoding
        try:
            osutils._cached_user_encoding = 'ascii'
            out, err = brz('log', encoding=encoding)
            if not fail:
                self.assertNotEqual(-1, out.find(self._message))
            else:
                self.assertNotEqual(-1, out.find('Message with ?'))
        finally:
            osutils._cached_user_encoding = old_encoding

    def test_log_handles_encoding(self):
        self.create_branch()
        for encoding in self.good_encodings:
            self.try_encoding(encoding)

    def test_log_handles_bad_encoding(self):
        self.create_branch()
        for encoding in self.bad_encodings:
            self.try_encoding(encoding, fail=True)

    def test_stdout_encoding(self):
        brz = self.run_bzr
        osutils._cached_user_encoding = 'cp1251'
        brz('init')
        self.build_tree(['a'])
        brz('add a')
        brz(['commit', '-m', 'Тест'])
        stdout, stderr = self.run_bzr_raw('log', encoding='cp866')
        message = stdout.splitlines()[-1]
        test_in_cp866 = b'\x92\xa5\xe1\xe2'
        test_in_cp1251 = b'\xd2\xe5\xf1\xf2'
        self.assertEqual(test_in_cp866, message[2:])
        self.assertEqual(-1, stdout.find(test_in_cp1251))