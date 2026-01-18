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
class HtdigestFileTest(TestCase):
    """test HtdigestFile class"""
    descriptionPrefix = 'HtdigestFile'
    sample_01 = b'user2:realm:549d2a5f4659ab39a80dac99e159ab19\nuser3:realm:a500bb8c02f6a9170ae46af10c898744\nuser4:realm:ab7b5d5f28ccc7666315f508c7358519\nuser1:realm:2a6cf53e7d8f8cf39d946dc880b14128\n'
    sample_02 = b'user3:realm:a500bb8c02f6a9170ae46af10c898744\nuser4:realm:ab7b5d5f28ccc7666315f508c7358519\n'
    sample_03 = b'user2:realm:5ba6d8328943c23c64b50f8b29566059\nuser3:realm:a500bb8c02f6a9170ae46af10c898744\nuser4:realm:ab7b5d5f28ccc7666315f508c7358519\nuser1:realm:2a6cf53e7d8f8cf39d946dc880b14128\nuser5:realm:03c55fdc6bf71552356ad401bdb9af19\n'
    sample_04_utf8 = b'user\xc3\xa6:realm\xc3\xa6:549d2a5f4659ab39a80dac99e159ab19\n'
    sample_04_latin1 = b'user\xe6:realm\xe6:549d2a5f4659ab39a80dac99e159ab19\n'

    def test_00_constructor_autoload(self):
        """test constructor autoload"""
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtdigestFile(path)
        self.assertEqual(ht.to_string(), self.sample_01)
        ht = apache.HtdigestFile(path, new=True)
        self.assertEqual(ht.to_string(), b'')
        os.remove(path)
        self.assertRaises(IOError, apache.HtdigestFile, path)

    def test_01_delete(self):
        """test delete()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertTrue(ht.delete('user1', 'realm'))
        self.assertTrue(ht.delete('user2', 'realm'))
        self.assertFalse(ht.delete('user5', 'realm'))
        self.assertFalse(ht.delete('user3', 'realm5'))
        self.assertEqual(ht.to_string(), self.sample_02)
        self.assertRaises(ValueError, ht.delete, 'user:', 'realm')
        self.assertRaises(ValueError, ht.delete, 'user', 'realm:')

    def test_01_delete_autosave(self):
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtdigestFile(path)
        self.assertTrue(ht.delete('user1', 'realm'))
        self.assertFalse(ht.delete('user3', 'realm5'))
        self.assertFalse(ht.delete('user5', 'realm'))
        self.assertEqual(get_file(path), self.sample_01)
        ht.autosave = True
        self.assertTrue(ht.delete('user2', 'realm'))
        self.assertEqual(get_file(path), self.sample_02)

    def test_02_set_password(self):
        """test update()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertTrue(ht.set_password('user2', 'realm', 'pass2x'))
        self.assertFalse(ht.set_password('user5', 'realm', 'pass5'))
        self.assertEqual(ht.to_string(), self.sample_03)
        self.assertRaises(TypeError, ht.set_password, 'user2', 'pass3')
        ht.default_realm = 'realm2'
        ht.set_password('user2', 'pass3')
        ht.check_password('user2', 'realm2', 'pass3')
        self.assertRaises(ValueError, ht.set_password, 'user:', 'realm', 'pass')
        self.assertRaises(ValueError, ht.set_password, 'u' * 256, 'realm', 'pass')
        self.assertRaises(ValueError, ht.set_password, 'user', 'realm:', 'pass')
        self.assertRaises(ValueError, ht.set_password, 'user', 'r' * 256, 'pass')
        with self.assertWarningList('update\\(\\) is deprecated'):
            ht.update('user2', 'realm2', 'test')
        self.assertTrue(ht.check_password('user2', 'test'))

    def test_03_users(self):
        """test users()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        ht.set_password('user5', 'realm', 'pass5')
        ht.delete('user3', 'realm')
        ht.set_password('user3', 'realm', 'pass3')
        self.assertEqual(sorted(ht.users('realm')), ['user1', 'user2', 'user3', 'user4', 'user5'])
        self.assertRaises(TypeError, ht.users, 1)

    def test_04_check_password(self):
        """test check_password()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertRaises(TypeError, ht.check_password, 1, 'realm', 'pass5')
        self.assertRaises(TypeError, ht.check_password, 'user', 1, 'pass5')
        self.assertIs(ht.check_password('user5', 'realm', 'pass5'), None)
        for i in irange(1, 5):
            i = str(i)
            self.assertTrue(ht.check_password('user' + i, 'realm', 'pass' + i))
            self.assertIs(ht.check_password('user' + i, 'realm', 'pass5'), False)
        self.assertRaises(TypeError, ht.check_password, 'user5', 'pass5')
        ht.default_realm = 'realm'
        self.assertTrue(ht.check_password('user1', 'pass1'))
        self.assertIs(ht.check_password('user5', 'pass5'), None)
        with self.assertWarningList(['verify\\(\\) is deprecated'] * 2):
            self.assertTrue(ht.verify('user1', 'realm', 'pass1'))
            self.assertFalse(ht.verify('user1', 'realm', 'pass2'))
        self.assertRaises(ValueError, ht.check_password, 'user:', 'realm', 'pass')

    def test_05_load(self):
        """test load()"""
        path = self.mktemp()
        set_file(path, '')
        backdate_file_mtime(path, 5)
        ha = apache.HtdigestFile(path)
        self.assertEqual(ha.to_string(), b'')
        ha.set_password('user1', 'realm', 'pass1')
        ha.load_if_changed()
        self.assertEqual(ha.to_string(), b'user1:realm:2a6cf53e7d8f8cf39d946dc880b14128\n')
        set_file(path, self.sample_01)
        ha.load_if_changed()
        self.assertEqual(ha.to_string(), self.sample_01)
        ha.set_password('user5', 'realm', 'pass5')
        ha.load()
        self.assertEqual(ha.to_string(), self.sample_01)
        hb = apache.HtdigestFile()
        self.assertRaises(RuntimeError, hb.load)
        self.assertRaises(RuntimeError, hb.load_if_changed)
        hc = apache.HtdigestFile()
        hc.load(path)
        self.assertEqual(hc.to_string(), self.sample_01)
        ensure_mtime_changed(path)
        set_file(path, '')
        with self.assertWarningList('load\\(force=False\\) is deprecated'):
            ha.load(force=False)
        self.assertEqual(ha.to_string(), b'')

    def test_06_save(self):
        """test save()"""
        path = self.mktemp()
        set_file(path, self.sample_01)
        ht = apache.HtdigestFile(path)
        ht.delete('user1', 'realm')
        ht.delete('user2', 'realm')
        ht.save()
        self.assertEqual(get_file(path), self.sample_02)
        hb = apache.HtdigestFile()
        hb.set_password('user1', 'realm', 'pass1')
        self.assertRaises(RuntimeError, hb.save)
        hb.save(path)
        self.assertEqual(get_file(path), hb.to_string())

    def test_07_realms(self):
        """test realms() & delete_realm()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertEqual(ht.delete_realm('x'), 0)
        self.assertEqual(ht.realms(), ['realm'])
        self.assertEqual(ht.delete_realm('realm'), 4)
        self.assertEqual(ht.realms(), [])
        self.assertEqual(ht.to_string(), b'')

    def test_08_get_hash(self):
        """test get_hash()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertEqual(ht.get_hash('user3', 'realm'), 'a500bb8c02f6a9170ae46af10c898744')
        self.assertEqual(ht.get_hash('user4', 'realm'), 'ab7b5d5f28ccc7666315f508c7358519')
        self.assertEqual(ht.get_hash('user5', 'realm'), None)
        with self.assertWarningList('find\\(\\) is deprecated'):
            self.assertEqual(ht.find('user4', 'realm'), 'ab7b5d5f28ccc7666315f508c7358519')

    def test_09_encodings(self):
        """test encoding parameter"""
        self.assertRaises(ValueError, apache.HtdigestFile, encoding='utf-16')
        ht = apache.HtdigestFile.from_string(self.sample_04_utf8, encoding='utf-8', return_unicode=True)
        self.assertEqual(ht.realms(), [u('realmæ')])
        self.assertEqual(ht.users(u('realmæ')), [u('useræ')])
        ht = apache.HtdigestFile.from_string(self.sample_04_latin1, encoding='latin-1', return_unicode=True)
        self.assertEqual(ht.realms(), [u('realmæ')])
        self.assertEqual(ht.users(u('realmæ')), [u('useræ')])

    def test_10_to_string(self):
        """test to_string()"""
        ht = apache.HtdigestFile.from_string(self.sample_01)
        self.assertEqual(ht.to_string(), self.sample_01)
        ht = apache.HtdigestFile()
        self.assertEqual(ht.to_string(), b'')

    def test_11_malformed(self):
        self.assertRaises(ValueError, apache.HtdigestFile.from_string, b'realm:user1:pass1:other\n')
        self.assertRaises(ValueError, apache.HtdigestFile.from_string, b'user1:pass1\n')