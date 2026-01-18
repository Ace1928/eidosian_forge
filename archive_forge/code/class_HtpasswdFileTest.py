from __future__ import with_statement
from logging import getLogger
import os
import subprocess
from passlib import apache, registry
from passlib.exc import MissingBackendError
from passlib.utils.compat import irange
from passlib.tests.backports import unittest
from passlib.tests.utils import TestCase, get_file, set_file, ensure_mtime_changed
from passlib.utils.compat import u
from passlib.utils import to_bytes
from passlib.utils.handlers import to_unicode_for_identify
class HtpasswdFileTest(TestCase):
    """test HtpasswdFile class"""
    descriptionPrefix = 'HtpasswdFile'
    sample_01 = b'user2:2CHkkwa2AtqGs\nuser3:{SHA}3ipNV1GrBtxPmHFC21fCbVCSXIo=\nuser4:pass4\nuser1:$apr1$t4tc7jTh$GPIWVUo8sQKJlUdV8V5vu0\n'
    sample_02 = b'user3:{SHA}3ipNV1GrBtxPmHFC21fCbVCSXIo=\nuser4:pass4\n'
    sample_03 = b'user2:pass2x\nuser3:{SHA}3ipNV1GrBtxPmHFC21fCbVCSXIo=\nuser4:pass4\nuser1:$apr1$t4tc7jTh$GPIWVUo8sQKJlUdV8V5vu0\nuser5:pass5\n'
    sample_04_utf8 = b'user\xc3\xa6:2CHkkwa2AtqGs\n'
    sample_04_latin1 = b'user\xe6:2CHkkwa2AtqGs\n'
    sample_dup = b'user1:pass1\nuser1:pass2\n'
    sample_05 = b'user2:2CHkkwa2AtqGs\nuser3:{SHA}3ipNV1GrBtxPmHFC21fCbVCSXIo=\nuser4:pass4\nuser1:$apr1$t4tc7jTh$GPIWVUo8sQKJlUdV8V5vu0\nuser5:$2a$12$yktDxraxijBZ360orOyCOePFGhuis/umyPNJoL5EbsLk.s6SWdrRO\nuser6:$5$rounds=110000$cCRp/xUUGVgwR4aP$p0.QKFS5qLNRqw1/47lXYiAcgIjJK.WjCO8nrEKuUK.\n'

    def test_00_constructor_autoload(self):
        """test constructor autoload"""
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtpasswdFile(path)
        self.assertEqual(ht.to_string(), self.sample_01)
        self.assertEqual(ht.path, path)
        self.assertTrue(ht.mtime)
        ht.path = path + 'x'
        self.assertEqual(ht.path, path + 'x')
        self.assertFalse(ht.mtime)
        ht = apache.HtpasswdFile(path, new=True)
        self.assertEqual(ht.to_string(), b'')
        self.assertEqual(ht.path, path)
        self.assertFalse(ht.mtime)
        with self.assertWarningList('``autoload=False`` is deprecated'):
            ht = apache.HtpasswdFile(path, autoload=False)
        self.assertEqual(ht.to_string(), b'')
        self.assertEqual(ht.path, path)
        self.assertFalse(ht.mtime)
        os.remove(path)
        self.assertRaises(IOError, apache.HtpasswdFile, path)

    def test_00_from_path(self):
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtpasswdFile.from_path(path)
        self.assertEqual(ht.to_string(), self.sample_01)
        self.assertEqual(ht.path, None)
        self.assertFalse(ht.mtime)

    def test_01_delete(self):
        """test delete()"""
        ht = apache.HtpasswdFile.from_string(self.sample_01)
        self.assertTrue(ht.delete('user1'))
        self.assertTrue(ht.delete('user2'))
        self.assertFalse(ht.delete('user5'))
        self.assertEqual(ht.to_string(), self.sample_02)
        self.assertRaises(ValueError, ht.delete, 'user:')

    def test_01_delete_autosave(self):
        path = self.mktemp()
        sample = b'user1:pass1\nuser2:pass2\n'
        set_file(path, sample)
        ht = apache.HtpasswdFile(path)
        ht.delete('user1')
        self.assertEqual(get_file(path), sample)
        ht = apache.HtpasswdFile(path, autosave=True)
        ht.delete('user1')
        self.assertEqual(get_file(path), b'user2:pass2\n')

    def test_02_set_password(self):
        """test set_password()"""
        ht = apache.HtpasswdFile.from_string(self.sample_01, default_scheme='plaintext')
        self.assertTrue(ht.set_password('user2', 'pass2x'))
        self.assertFalse(ht.set_password('user5', 'pass5'))
        self.assertEqual(ht.to_string(), self.sample_03)
        with self.assertWarningList('``default`` is deprecated'):
            ht = apache.HtpasswdFile.from_string(self.sample_01, default='plaintext')
        self.assertTrue(ht.set_password('user2', 'pass2x'))
        self.assertFalse(ht.set_password('user5', 'pass5'))
        self.assertEqual(ht.to_string(), self.sample_03)
        self.assertRaises(ValueError, ht.set_password, 'user:', 'pass')
        with self.assertWarningList('update\\(\\) is deprecated'):
            ht.update('user2', 'test')
        self.assertTrue(ht.check_password('user2', 'test'))

    def test_02_set_password_autosave(self):
        path = self.mktemp()
        sample = b'user1:pass1\n'
        set_file(path, sample)
        ht = apache.HtpasswdFile(path)
        ht.set_password('user1', 'pass2')
        self.assertEqual(get_file(path), sample)
        ht = apache.HtpasswdFile(path, default_scheme='plaintext', autosave=True)
        ht.set_password('user1', 'pass2')
        self.assertEqual(get_file(path), b'user1:pass2\n')

    def test_02_set_password_default_scheme(self):
        """test set_password() -- default_scheme"""

        def check(scheme):
            ht = apache.HtpasswdFile(default_scheme=scheme)
            ht.set_password('user1', 'pass1')
            return ht.context.identify(ht.get_hash('user1'))
        self.assertEqual(check('sha256_crypt'), 'sha256_crypt')
        self.assertEqual(check('des_crypt'), 'des_crypt')
        self.assertRaises(KeyError, check, 'xxx')
        self.assertEqual(check('portable'), apache.htpasswd_defaults['portable'])
        self.assertEqual(check('portable_apache_22'), apache.htpasswd_defaults['portable_apache_22'])
        self.assertEqual(check('host_apache_22'), apache.htpasswd_defaults['host_apache_22'])
        self.assertEqual(check(None), apache.htpasswd_defaults['portable_apache_22'])

    def test_03_users(self):
        """test users()"""
        ht = apache.HtpasswdFile.from_string(self.sample_01)
        ht.set_password('user5', 'pass5')
        ht.delete('user3')
        ht.set_password('user3', 'pass3')
        self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user3', 'user4', 'user5'])

    def test_04_check_password(self):
        """test check_password()"""
        ht = apache.HtpasswdFile.from_string(self.sample_05)
        self.assertRaises(TypeError, ht.check_password, 1, 'pass9')
        self.assertTrue(ht.check_password('user9', 'pass9') is None)
        for i in irange(1, 7):
            i = str(i)
            try:
                self.assertTrue(ht.check_password('user' + i, 'pass' + i))
                self.assertTrue(ht.check_password('user' + i, 'pass9') is False)
            except MissingBackendError:
                if i == '5':
                    continue
                raise
        self.assertRaises(ValueError, ht.check_password, 'user:', 'pass')
        with self.assertWarningList(['verify\\(\\) is deprecated'] * 2):
            self.assertTrue(ht.verify('user1', 'pass1'))
            self.assertFalse(ht.verify('user1', 'pass2'))

    def test_05_load(self):
        """test load()"""
        path = self.mktemp()
        set_file(path, '')
        backdate_file_mtime(path, 5)
        ha = apache.HtpasswdFile(path, default_scheme='plaintext')
        self.assertEqual(ha.to_string(), b'')
        ha.set_password('user1', 'pass1')
        ha.load_if_changed()
        self.assertEqual(ha.to_string(), b'user1:pass1\n')
        set_file(path, self.sample_01)
        ha.load_if_changed()
        self.assertEqual(ha.to_string(), self.sample_01)
        ha.set_password('user5', 'pass5')
        ha.load()
        self.assertEqual(ha.to_string(), self.sample_01)
        hb = apache.HtpasswdFile()
        self.assertRaises(RuntimeError, hb.load)
        self.assertRaises(RuntimeError, hb.load_if_changed)
        set_file(path, self.sample_dup)
        hc = apache.HtpasswdFile()
        hc.load(path)
        self.assertTrue(hc.check_password('user1', 'pass1'))

    def test_06_save(self):
        """test save()"""
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtpasswdFile(path)
        ht.delete('user1')
        ht.delete('user2')
        ht.save()
        self.assertEqual(get_file(path), self.sample_02)
        hb = apache.HtpasswdFile(default_scheme='plaintext')
        hb.set_password('user1', 'pass1')
        self.assertRaises(RuntimeError, hb.save)
        hb.save(path)
        self.assertEqual(get_file(path), b'user1:pass1\n')

    def test_07_encodings(self):
        """test 'encoding' kwd"""
        self.assertRaises(ValueError, apache.HtpasswdFile, encoding='utf-16')
        ht = apache.HtpasswdFile.from_string(self.sample_04_utf8, encoding='utf-8', return_unicode=True)
        self.assertEqual(ht.users(), [u('useræ')])
        with self.assertWarningList('``encoding=None`` is deprecated'):
            ht = apache.HtpasswdFile.from_string(self.sample_04_utf8, encoding=None)
        self.assertEqual(ht.users(), [b'user\xc3\xa6'])
        ht = apache.HtpasswdFile.from_string(self.sample_04_latin1, encoding='latin-1', return_unicode=True)
        self.assertEqual(ht.users(), [u('useræ')])

    def test_08_get_hash(self):
        """test get_hash()"""
        ht = apache.HtpasswdFile.from_string(self.sample_01)
        self.assertEqual(ht.get_hash('user3'), b'{SHA}3ipNV1GrBtxPmHFC21fCbVCSXIo=')
        self.assertEqual(ht.get_hash('user4'), b'pass4')
        self.assertEqual(ht.get_hash('user5'), None)
        with self.assertWarningList('find\\(\\) is deprecated'):
            self.assertEqual(ht.find('user4'), b'pass4')

    def test_09_to_string(self):
        """test to_string"""
        ht = apache.HtpasswdFile.from_string(self.sample_01)
        self.assertEqual(ht.to_string(), self.sample_01)
        ht = apache.HtpasswdFile()
        self.assertEqual(ht.to_string(), b'')

    def test_10_repr(self):
        ht = apache.HtpasswdFile('fakepath', autosave=True, new=True, encoding='latin-1')
        repr(ht)

    def test_11_malformed(self):
        self.assertRaises(ValueError, apache.HtpasswdFile.from_string, b'realm:user1:pass1\n')
        self.assertRaises(ValueError, apache.HtpasswdFile.from_string, b'pass1\n')

    def test_12_from_string(self):
        self.assertRaises(TypeError, apache.HtpasswdFile.from_string, b'', path=None)

    def test_13_whitespace(self):
        """whitespace & comment handling"""
        source = to_bytes('\nuser2:pass2\nuser4:pass4\nuser7:pass7\r\n \t \nuser1:pass1\n # legacy users\n#user6:pass6\nuser5:pass5\n\n')
        ht = apache.HtpasswdFile.from_string(source)
        self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user7'])
        ht.set_hash('user4', 'althash4')
        self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user7'])
        ht.set_hash('user6', 'althash6')
        self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user6', 'user7'])
        ht.delete('user7')
        self.assertEqual(sorted(ht.users()), ['user1', 'user2', 'user4', 'user5', 'user6'])
        target = to_bytes('\nuser2:pass2\nuser4:althash4\n \t \nuser1:pass1\n # legacy users\n#user6:pass6\nuser5:pass5\nuser6:althash6\n')
        self.assertEqual(ht.to_string(), target)

    @requires_htpasswd_cmd
    def test_htpasswd_cmd_verify(self):
        """
        verify "htpasswd" command can read output
        """
        path = self.mktemp()
        ht = apache.HtpasswdFile(path=path, new=True)

        def hash_scheme(pwd, scheme):
            return ht.context.handler(scheme).hash(pwd)
        ht.set_hash('user1', hash_scheme('password', 'apr_md5_crypt'))
        host_no_bcrypt = apache.htpasswd_defaults['host_apache_22']
        ht.set_hash('user2', hash_scheme('password', host_no_bcrypt))
        host_best = apache.htpasswd_defaults['host']
        ht.set_hash('user3', hash_scheme('password', host_best))
        ht.set_hash('user4', '$xxx$foo$bar$baz')
        ht.save()
        self.assertFalse(_call_htpasswd_verify(path, 'user1', 'wrong'))
        self.assertFalse(_call_htpasswd_verify(path, 'user2', 'wrong'))
        self.assertFalse(_call_htpasswd_verify(path, 'user3', 'wrong'))
        self.assertFalse(_call_htpasswd_verify(path, 'user4', 'wrong'))
        self.assertTrue(_call_htpasswd_verify(path, 'user1', 'password'))
        self.assertTrue(_call_htpasswd_verify(path, 'user2', 'password'))
        self.assertTrue(_call_htpasswd_verify(path, 'user3', 'password'))

    @requires_htpasswd_cmd
    @unittest.skipUnless(registry.has_backend('bcrypt'), 'bcrypt support required')
    def test_htpasswd_cmd_verify_bcrypt(self):
        """
        verify "htpasswd" command can read bcrypt format

        this tests for regression of issue 95, where we output "$2b$" instead of "$2y$";
        fixed in v1.7.2.
        """
        path = self.mktemp()
        ht = apache.HtpasswdFile(path=path, new=True)

        def hash_scheme(pwd, scheme):
            return ht.context.handler(scheme).hash(pwd)
        ht.set_hash('user1', hash_scheme('password', 'bcrypt'))
        ht.save()
        self.assertFalse(_call_htpasswd_verify(path, 'user1', 'wrong'))
        if HAVE_HTPASSWD_BCRYPT:
            self.assertTrue(_call_htpasswd_verify(path, 'user1', 'password'))
        else:
            self.assertFalse(_call_htpasswd_verify(path, 'user1', 'password'))