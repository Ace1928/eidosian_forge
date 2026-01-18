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
class LazyCryptContextTest(TestCase):
    descriptionPrefix = 'LazyCryptContext'

    def setUp(self):
        TestCase.setUp(self)
        unload_handler_name('dummy_2')
        self.addCleanup(unload_handler_name, 'dummy_2')
        warnings.filterwarnings('ignore', 'CryptContext\\(\\)\\.replace\\(\\) has been deprecated')
        warnings.filterwarnings('ignore', '.*(CryptPolicy|context\\.policy).*(has|have) been deprecated.*')

    def test_kwd_constructor(self):
        """test plain kwds"""
        self.assertFalse(has_crypt_handler('dummy_2'))
        register_crypt_handler_path('dummy_2', 'passlib.tests.test_context')
        cc = LazyCryptContext(iter(['dummy_2', 'des_crypt']), deprecated=['des_crypt'])
        self.assertFalse(has_crypt_handler('dummy_2', True))
        self.assertTrue(cc.policy.handler_is_deprecated('des_crypt'))
        self.assertEqual(cc.policy.schemes(), ['dummy_2', 'des_crypt'])
        self.assertTrue(has_crypt_handler('dummy_2', True))

    def test_callable_constructor(self):
        """test create_policy() hook, returning CryptPolicy"""
        self.assertFalse(has_crypt_handler('dummy_2'))
        register_crypt_handler_path('dummy_2', 'passlib.tests.test_context')

        def create_policy(flag=False):
            self.assertTrue(flag)
            return CryptPolicy(schemes=iter(['dummy_2', 'des_crypt']), deprecated=['des_crypt'])
        cc = LazyCryptContext(create_policy=create_policy, flag=True)
        self.assertFalse(has_crypt_handler('dummy_2', True))
        self.assertTrue(cc.policy.handler_is_deprecated('des_crypt'))
        self.assertEqual(cc.policy.schemes(), ['dummy_2', 'des_crypt'])
        self.assertTrue(has_crypt_handler('dummy_2', True))