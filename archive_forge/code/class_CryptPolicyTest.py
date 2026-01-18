from __future__ import with_statement
from logging import getLogger
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, CryptPolicy, LazyCryptContext
from passlib.utils import to_bytes, to_unicode
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase, set_file
from passlib.registry import (register_crypt_handler_path,
class CryptPolicyTest(TestCase):
    """test CryptPolicy object"""
    descriptionPrefix = 'CryptPolicy'
    sample_config_1s = '[passlib]\nschemes = des_crypt, md5_crypt, bsdi_crypt, sha512_crypt\ndefault = md5_crypt\nall.vary_rounds = 10%%\nbsdi_crypt.max_rounds = 30000\nbsdi_crypt.default_rounds = 25000\nsha512_crypt.max_rounds = 50000\nsha512_crypt.min_rounds = 40000\n'
    sample_config_1s_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sample_config_1s.cfg'))
    if not os.path.exists(sample_config_1s_path) and resource_filename:
        sample_config_1s_path = resource_filename('passlib.tests', 'sample_config_1s.cfg')
    assert sample_config_1s.startswith('[passlib]\nschemes')
    sample_config_1pd = dict(schemes=['des_crypt', 'md5_crypt', 'bsdi_crypt', 'sha512_crypt'], default='md5_crypt', all__vary_rounds=0.1, bsdi_crypt__max_rounds=30000, bsdi_crypt__default_rounds=25000, sha512_crypt__max_rounds=50000, sha512_crypt__min_rounds=40000)
    sample_config_1pid = {'schemes': 'des_crypt, md5_crypt, bsdi_crypt, sha512_crypt', 'default': 'md5_crypt', 'all.vary_rounds': 0.1, 'bsdi_crypt.max_rounds': 30000, 'bsdi_crypt.default_rounds': 25000, 'sha512_crypt.max_rounds': 50000, 'sha512_crypt.min_rounds': 40000}
    sample_config_1prd = dict(schemes=[hash.des_crypt, hash.md5_crypt, hash.bsdi_crypt, hash.sha512_crypt], default='md5_crypt', all__vary_rounds=0.1, bsdi_crypt__max_rounds=30000, bsdi_crypt__default_rounds=25000, sha512_crypt__max_rounds=50000, sha512_crypt__min_rounds=40000)
    sample_config_2s = '[passlib]\nbsdi_crypt.min_rounds = 29000\nbsdi_crypt.max_rounds = 35000\nbsdi_crypt.default_rounds = 31000\nsha512_crypt.min_rounds = 45000\n'
    sample_config_2pd = dict(bsdi_crypt__min_rounds=29000, bsdi_crypt__max_rounds=35000, bsdi_crypt__default_rounds=31000, sha512_crypt__min_rounds=45000)
    sample_config_12pd = dict(schemes=['des_crypt', 'md5_crypt', 'bsdi_crypt', 'sha512_crypt'], default='md5_crypt', all__vary_rounds=0.1, bsdi_crypt__min_rounds=29000, bsdi_crypt__max_rounds=35000, bsdi_crypt__default_rounds=31000, sha512_crypt__max_rounds=50000, sha512_crypt__min_rounds=45000)
    sample_config_3pd = dict(default='sha512_crypt')
    sample_config_123pd = dict(schemes=['des_crypt', 'md5_crypt', 'bsdi_crypt', 'sha512_crypt'], default='sha512_crypt', all__vary_rounds=0.1, bsdi_crypt__min_rounds=29000, bsdi_crypt__max_rounds=35000, bsdi_crypt__default_rounds=31000, sha512_crypt__max_rounds=50000, sha512_crypt__min_rounds=45000)
    sample_config_4s = '\n[passlib]\nschemes = sha512_crypt\nall.vary_rounds = 10%%\ndefault.sha512_crypt.max_rounds = 20000\nadmin.all.vary_rounds = 5%%\nadmin.sha512_crypt.max_rounds = 40000\n'
    sample_config_4pd = dict(schemes=['sha512_crypt'], all__vary_rounds=0.1, sha512_crypt__max_rounds=20000, admin__all__vary_rounds=0.05, admin__sha512_crypt__max_rounds=40000)
    sample_config_5s = sample_config_1s + 'deprecated = des_crypt\nadmin__context__deprecated = des_crypt, bsdi_crypt\n'
    sample_config_5pd = sample_config_1pd.copy()
    sample_config_5pd.update(deprecated=['des_crypt'], admin__context__deprecated=['des_crypt', 'bsdi_crypt'])
    sample_config_5pid = sample_config_1pid.copy()
    sample_config_5pid.update({'deprecated': 'des_crypt', 'admin.context.deprecated': 'des_crypt, bsdi_crypt'})
    sample_config_5prd = sample_config_1prd.copy()
    sample_config_5prd.update({'deprecated': ['des_crypt'], 'admin__context__deprecated': ['des_crypt', 'bsdi_crypt']})

    def setUp(self):
        TestCase.setUp(self)
        warnings.filterwarnings('ignore', 'The CryptPolicy class has been deprecated')
        warnings.filterwarnings('ignore', 'the method.*hash_needs_update.*is deprecated')
        warnings.filterwarnings('ignore', "The 'all' scheme is deprecated.*")
        warnings.filterwarnings('ignore', 'bsdi_crypt rounds should be odd')

    def test_00_constructor(self):
        """test CryptPolicy() constructor"""
        policy = CryptPolicy(**self.sample_config_1pd)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy(self.sample_config_1pd)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        self.assertRaises(TypeError, CryptPolicy, {}, {})
        self.assertRaises(TypeError, CryptPolicy, {}, dummy=1)
        self.assertRaises(TypeError, CryptPolicy, schemes=['des_crypt', 'md5_crypt', 'bsdi_crypt', 'sha512_crypt'], bad__key__bsdi_crypt__max_rounds=30000)

        class nameless(uh.StaticHandler):
            name = None
        self.assertRaises(ValueError, CryptPolicy, schemes=[nameless])
        self.assertRaises(TypeError, CryptPolicy, schemes=[uh.StaticHandler])

        class dummy_1(uh.StaticHandler):
            name = 'dummy_1'
        self.assertRaises(KeyError, CryptPolicy, schemes=[dummy_1, dummy_1])
        self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], deprecated=['md5_crypt'])
        self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], default='md5_crypt')

    def test_01_from_path_simple(self):
        """test CryptPolicy.from_path() constructor"""
        path = self.sample_config_1s_path
        policy = CryptPolicy.from_path(path)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        self.assertRaises(EnvironmentError, CryptPolicy.from_path, path + 'xxx')

    def test_01_from_path(self):
        """test CryptPolicy.from_path() constructor with encodings"""
        path = self.mktemp()
        set_file(path, self.sample_config_1s)
        policy = CryptPolicy.from_path(path)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        set_file(path, self.sample_config_1s.replace('\n', '\r\n'))
        policy = CryptPolicy.from_path(path)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        uc2 = to_bytes(self.sample_config_1s, 'utf-16', source_encoding='utf-8')
        set_file(path, uc2)
        policy = CryptPolicy.from_path(path, encoding='utf-16')
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)

    def test_02_from_string(self):
        """test CryptPolicy.from_string() constructor"""
        policy = CryptPolicy.from_string(self.sample_config_1s)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy.from_string(self.sample_config_1s.replace('\n', '\r\n'))
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        data = to_unicode(self.sample_config_1s)
        policy = CryptPolicy.from_string(data)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        uc2 = to_bytes(self.sample_config_1s, 'utf-16', source_encoding='utf-8')
        policy = CryptPolicy.from_string(uc2, encoding='utf-16')
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy.from_string(self.sample_config_4s)
        self.assertEqual(policy.to_dict(), self.sample_config_4pd)

    def test_03_from_source(self):
        """test CryptPolicy.from_source() constructor"""
        policy = CryptPolicy.from_source(self.sample_config_1s_path)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy.from_source(self.sample_config_1s)
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy.from_source(self.sample_config_1pd.copy())
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        p2 = CryptPolicy.from_source(policy)
        self.assertIs(policy, p2)
        self.assertRaises(TypeError, CryptPolicy.from_source, 1)
        self.assertRaises(TypeError, CryptPolicy.from_source, [])

    def test_04_from_sources(self):
        """test CryptPolicy.from_sources() constructor"""
        self.assertRaises(ValueError, CryptPolicy.from_sources, [])
        policy = CryptPolicy.from_sources([self.sample_config_1s])
        self.assertEqual(policy.to_dict(), self.sample_config_1pd)
        policy = CryptPolicy.from_sources([self.sample_config_1s_path, self.sample_config_2s, self.sample_config_3pd])
        self.assertEqual(policy.to_dict(), self.sample_config_123pd)

    def test_05_replace(self):
        """test CryptPolicy.replace() constructor"""
        p1 = CryptPolicy(**self.sample_config_1pd)
        p2 = p1.replace(**self.sample_config_2pd)
        self.assertEqual(p2.to_dict(), self.sample_config_12pd)
        p2b = p2.replace(**self.sample_config_2pd)
        self.assertEqual(p2b.to_dict(), self.sample_config_12pd)
        p3 = p2.replace(self.sample_config_3pd)
        self.assertEqual(p3.to_dict(), self.sample_config_123pd)

    def test_06_forbidden(self):
        """test CryptPolicy() forbidden kwds"""
        self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], des_crypt__salt='xx')
        self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], all__salt='xx')
        self.assertRaises(KeyError, CryptPolicy, schemes=['des_crypt'], user__context__schemes=['md5_crypt'])

    def test_10_has_schemes(self):
        """test has_schemes() method"""
        p1 = CryptPolicy(**self.sample_config_1pd)
        self.assertTrue(p1.has_schemes())
        p3 = CryptPolicy(**self.sample_config_3pd)
        self.assertTrue(not p3.has_schemes())

    def test_11_iter_handlers(self):
        """test iter_handlers() method"""
        p1 = CryptPolicy(**self.sample_config_1pd)
        s = self.sample_config_1prd['schemes']
        self.assertEqual(list(p1.iter_handlers()), s)
        p3 = CryptPolicy(**self.sample_config_3pd)
        self.assertEqual(list(p3.iter_handlers()), [])

    def test_12_get_handler(self):
        """test get_handler() method"""
        p1 = CryptPolicy(**self.sample_config_1pd)
        self.assertIs(p1.get_handler('bsdi_crypt'), hash.bsdi_crypt)
        self.assertIs(p1.get_handler('sha256_crypt'), None)
        self.assertRaises(KeyError, p1.get_handler, 'sha256_crypt', required=True)
        self.assertIs(p1.get_handler(), hash.md5_crypt)

    def test_13_get_options(self):
        """test get_options() method"""
        p12 = CryptPolicy(**self.sample_config_12pd)
        self.assertEqual(p12.get_options('bsdi_crypt'), dict(vary_rounds=0.1, min_rounds=29000, max_rounds=35000, default_rounds=31000))
        self.assertEqual(p12.get_options('sha512_crypt'), dict(vary_rounds=0.1, min_rounds=45000, max_rounds=50000))
        p4 = CryptPolicy.from_string(self.sample_config_4s)
        self.assertEqual(p4.get_options('sha512_crypt'), dict(vary_rounds=0.1, max_rounds=20000))
        self.assertEqual(p4.get_options('sha512_crypt', 'user'), dict(vary_rounds=0.1, max_rounds=20000))
        self.assertEqual(p4.get_options('sha512_crypt', 'admin'), dict(vary_rounds=0.05, max_rounds=40000))

    def test_14_handler_is_deprecated(self):
        """test handler_is_deprecated() method"""
        pa = CryptPolicy(**self.sample_config_1pd)
        pb = CryptPolicy(**self.sample_config_5pd)
        self.assertFalse(pa.handler_is_deprecated('des_crypt'))
        self.assertFalse(pa.handler_is_deprecated(hash.bsdi_crypt))
        self.assertFalse(pa.handler_is_deprecated('sha512_crypt'))
        self.assertTrue(pb.handler_is_deprecated('des_crypt'))
        self.assertFalse(pb.handler_is_deprecated(hash.bsdi_crypt))
        self.assertFalse(pb.handler_is_deprecated('sha512_crypt'))
        self.assertTrue(pb.handler_is_deprecated('des_crypt', 'user'))
        self.assertFalse(pb.handler_is_deprecated('bsdi_crypt', 'user'))
        self.assertTrue(pb.handler_is_deprecated('des_crypt', 'admin'))
        self.assertTrue(pb.handler_is_deprecated('bsdi_crypt', 'admin'))
        pc = CryptPolicy(schemes=['md5_crypt', 'des_crypt'], deprecated=['md5_crypt'], user__context__deprecated=['des_crypt'])
        self.assertTrue(pc.handler_is_deprecated('md5_crypt'))
        self.assertFalse(pc.handler_is_deprecated('des_crypt'))
        self.assertFalse(pc.handler_is_deprecated('md5_crypt', 'user'))
        self.assertTrue(pc.handler_is_deprecated('des_crypt', 'user'))

    def test_15_min_verify_time(self):
        """test get_min_verify_time() method"""
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        pa = CryptPolicy()
        self.assertEqual(pa.get_min_verify_time(), 0)
        self.assertEqual(pa.get_min_verify_time('admin'), 0)
        pb = pa.replace(min_verify_time=0.1)
        self.assertEqual(pb.get_min_verify_time(), 0)
        self.assertEqual(pb.get_min_verify_time('admin'), 0)

    def test_20_iter_config(self):
        """test iter_config() method"""
        p5 = CryptPolicy(**self.sample_config_5pd)
        self.assertEqual(dict(p5.iter_config()), self.sample_config_5pd)
        self.assertEqual(dict(p5.iter_config(resolve=True)), self.sample_config_5prd)
        self.assertEqual(dict(p5.iter_config(ini=True)), self.sample_config_5pid)

    def test_21_to_dict(self):
        """test to_dict() method"""
        p5 = CryptPolicy(**self.sample_config_5pd)
        self.assertEqual(p5.to_dict(), self.sample_config_5pd)
        self.assertEqual(p5.to_dict(resolve=True), self.sample_config_5prd)

    def test_22_to_string(self):
        """test to_string() method"""
        pa = CryptPolicy(**self.sample_config_5pd)
        s = pa.to_string()
        pb = CryptPolicy.from_string(s)
        self.assertEqual(pb.to_dict(), self.sample_config_5pd)
        s = pa.to_string(encoding='latin-1')
        self.assertIsInstance(s, bytes)