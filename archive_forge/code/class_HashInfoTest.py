from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
class HashInfoTest(TestCase):
    """test various crypto functions"""
    descriptionPrefix = 'passlib.crypto.digest'
    norm_hash_formats = ['hashlib', 'iana']
    norm_hash_samples = [('md5', 'md5', 'SCRAM-MD5-PLUS', 'MD-5'), ('sha1', 'sha-1', 'SCRAM-SHA-1', 'SHA1'), ('sha256', 'sha-256', 'SHA_256', 'sha2-256'), ('ripemd160', 'ripemd-160', 'SCRAM-RIPEMD-160', 'RIPEmd160', 'ripemd', 'SCRAM-RIPEMD'), ('sha4_256', 'sha4-256', 'SHA4-256', 'SHA-4-256'), ('test128', 'test-128', 'TEST128'), ('test2', 'test2', 'TEST-2'), ('test3_128', 'test3-128', 'TEST-3-128')]

    def test_norm_hash_name(self):
        """norm_hash_name()"""
        from itertools import chain
        from passlib.crypto.digest import norm_hash_name, _known_hash_names
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__)
        warnings.filterwarnings('ignore', '.*unknown hash')
        warnings.filterwarnings('ignore', '.*unsupported hash')
        self.assertEqual(norm_hash_name(u('MD4')), 'md4')
        self.assertEqual(norm_hash_name(b'MD4'), 'md4')
        self.assertRaises(TypeError, norm_hash_name, None)
        for row in chain(_known_hash_names, self.norm_hash_samples):
            for idx, format in enumerate(self.norm_hash_formats):
                correct = row[idx]
                for value in row:
                    result = norm_hash_name(value, format)
                    self.assertEqual(result, correct, 'name=%r, format=%r:' % (value, format))

    def test_lookup_hash_ctor(self):
        """lookup_hash() -- constructor"""
        from passlib.crypto.digest import lookup_hash
        self.assertRaises(ValueError, lookup_hash, 'new')
        self.assertRaises(ValueError, lookup_hash, '__name__')
        self.assertRaises(ValueError, lookup_hash, 'sha4')
        self.assertEqual(lookup_hash('md5'), (hashlib.md5, 16, 64))
        try:
            hashlib.new('sha')
            has_sha = True
        except ValueError:
            has_sha = False
        if has_sha:
            record = lookup_hash('sha')
            const = record[0]
            self.assertEqual(record, (const, 20, 64))
            self.assertEqual(hexlify(const(b'abc').digest()), b'0164b8a914cd2a5e74c4f7ff082c4d97f1edf880')
        else:
            self.assertRaises(ValueError, lookup_hash, 'sha')
        try:
            hashlib.new('md4')
            has_md4 = True
        except ValueError:
            has_md4 = False
        record = lookup_hash('md4')
        const = record[0]
        if not has_md4:
            from passlib.crypto._md4 import md4
            self.assertIs(const, md4)
        self.assertEqual(record, (const, 16, 64))
        self.assertEqual(hexlify(const(b'abc').digest()), b'a448017aaf21d8525fc10ae87aa6729d')
        self.assertIs(lookup_hash('md5'), lookup_hash('md5'))

    def test_lookup_hash_w_unknown_name(self):
        """lookup_hash() -- unknown hash name"""
        from passlib.crypto.digest import lookup_hash
        self.assertRaises(UnknownHashError, lookup_hash, 'xxx256')
        info = lookup_hash('xxx256', required=False)
        self.assertFalse(info.supported)
        self.assertRaisesRegex(UnknownHashError, "unknown hash: 'xxx256'", info.const)
        self.assertEqual(info.name, 'xxx256')
        self.assertEqual(info.digest_size, None)
        self.assertEqual(info.block_size, None)
        info2 = lookup_hash('xxx256', required=False)
        self.assertIs(info2, info)

    def test_mock_fips_mode(self):
        """
        lookup_hash() -- test set_mock_fips_mode()
        """
        from passlib.crypto.digest import lookup_hash, _set_mock_fips_mode
        if not lookup_hash('md5', required=False).supported:
            raise self.skipTest('md5 not supported')
        _set_mock_fips_mode()
        self.addCleanup(_set_mock_fips_mode, False)
        pat = "'md5' hash disabled for fips"
        self.assertRaisesRegex(UnknownHashError, pat, lookup_hash, 'md5')
        info = lookup_hash('md5', required=False)
        self.assertRegex(info.error_text, pat)
        self.assertRaisesRegex(UnknownHashError, pat, info.const)
        self.assertEqual(info.digest_size, 16)
        self.assertEqual(info.block_size, 64)

    def test_lookup_hash_metadata(self):
        """lookup_hash() -- metadata"""
        from passlib.crypto.digest import lookup_hash
        info = lookup_hash('sha256')
        self.assertEqual(info.name, 'sha256')
        self.assertEqual(info.iana_name, 'sha-256')
        self.assertEqual(info.block_size, 64)
        self.assertEqual(info.digest_size, 32)
        self.assertIs(lookup_hash('SHA2-256'), info)
        info = lookup_hash('md5')
        self.assertEqual(info.name, 'md5')
        self.assertEqual(info.iana_name, 'md5')
        self.assertEqual(info.block_size, 64)
        self.assertEqual(info.digest_size, 16)

    def test_lookup_hash_alt_types(self):
        """lookup_hash() -- alternate types"""
        from passlib.crypto.digest import lookup_hash
        info = lookup_hash('sha256')
        self.assertIs(lookup_hash(info), info)
        self.assertIs(lookup_hash(info.const), info)
        self.assertRaises(TypeError, lookup_hash, 123)