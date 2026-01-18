from io import StringIO
import re
import sys
import datetime
import unittest
import tornado
from tornado.escape import utf8
from tornado.util import (
import typing
from typing import cast
class ConfigurableTest(unittest.TestCase):

    def setUp(self):
        self.saved = TestConfigurable._save_configuration()
        self.saved3 = TestConfig3._save_configuration()

    def tearDown(self):
        TestConfigurable._restore_configuration(self.saved)
        TestConfig3._restore_configuration(self.saved3)

    def checkSubclasses(self):
        self.assertIsInstance(TestConfig1(), TestConfig1)
        self.assertIsInstance(TestConfig2(), TestConfig2)
        obj = TestConfig1(a=1)
        self.assertEqual(obj.a, 1)
        obj2 = TestConfig2(b=2)
        self.assertEqual(obj2.b, 2)

    def test_default(self):
        obj = cast(TestConfig1, TestConfigurable())
        self.assertIsInstance(obj, TestConfig1)
        self.assertIs(obj.a, None)
        obj = cast(TestConfig1, TestConfigurable(a=1))
        self.assertIsInstance(obj, TestConfig1)
        self.assertEqual(obj.a, 1)
        self.checkSubclasses()

    def test_config_class(self):
        TestConfigurable.configure(TestConfig2)
        obj = cast(TestConfig2, TestConfigurable())
        self.assertIsInstance(obj, TestConfig2)
        self.assertIs(obj.b, None)
        obj = cast(TestConfig2, TestConfigurable(b=2))
        self.assertIsInstance(obj, TestConfig2)
        self.assertEqual(obj.b, 2)
        self.checkSubclasses()

    def test_config_str(self):
        TestConfigurable.configure('tornado.test.util_test.TestConfig2')
        obj = cast(TestConfig2, TestConfigurable())
        self.assertIsInstance(obj, TestConfig2)
        self.assertIs(obj.b, None)
        obj = cast(TestConfig2, TestConfigurable(b=2))
        self.assertIsInstance(obj, TestConfig2)
        self.assertEqual(obj.b, 2)
        self.checkSubclasses()

    def test_config_args(self):
        TestConfigurable.configure(None, a=3)
        obj = cast(TestConfig1, TestConfigurable())
        self.assertIsInstance(obj, TestConfig1)
        self.assertEqual(obj.a, 3)
        obj = cast(TestConfig1, TestConfigurable(42, a=4))
        self.assertIsInstance(obj, TestConfig1)
        self.assertEqual(obj.a, 4)
        self.assertEqual(obj.pos_arg, 42)
        self.checkSubclasses()
        obj = TestConfig1()
        self.assertIs(obj.a, None)

    def test_config_class_args(self):
        TestConfigurable.configure(TestConfig2, b=5)
        obj = cast(TestConfig2, TestConfigurable())
        self.assertIsInstance(obj, TestConfig2)
        self.assertEqual(obj.b, 5)
        obj = cast(TestConfig2, TestConfigurable(42, b=6))
        self.assertIsInstance(obj, TestConfig2)
        self.assertEqual(obj.b, 6)
        self.assertEqual(obj.pos_arg, 42)
        self.checkSubclasses()
        obj = TestConfig2()
        self.assertIs(obj.b, None)

    def test_config_multi_level(self):
        TestConfigurable.configure(TestConfig3, a=1)
        obj = cast(TestConfig3A, TestConfigurable())
        self.assertIsInstance(obj, TestConfig3A)
        self.assertEqual(obj.a, 1)
        TestConfigurable.configure(TestConfig3)
        TestConfig3.configure(TestConfig3B, b=2)
        obj2 = cast(TestConfig3B, TestConfigurable())
        self.assertIsInstance(obj2, TestConfig3B)
        self.assertEqual(obj2.b, 2)

    def test_config_inner_level(self):
        obj = TestConfig3()
        self.assertIsInstance(obj, TestConfig3A)
        TestConfig3.configure(TestConfig3B)
        obj = TestConfig3()
        self.assertIsInstance(obj, TestConfig3B)
        obj2 = TestConfigurable()
        self.assertIsInstance(obj2, TestConfig1)
        TestConfigurable.configure(TestConfig2)
        obj3 = TestConfigurable()
        self.assertIsInstance(obj3, TestConfig2)
        obj = TestConfig3()
        self.assertIsInstance(obj, TestConfig3B)