import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class StrUtilsTest(test_base.BaseTestCase):

    def test_bool_bool_from_string(self):
        self.assertTrue(strutils.bool_from_string(True))
        self.assertFalse(strutils.bool_from_string(False))

    def test_bool_bool_from_string_default(self):
        self.assertTrue(strutils.bool_from_string('', default=True))
        self.assertFalse(strutils.bool_from_string('wibble', default=False))

    def _test_bool_from_string(self, c):
        self.assertTrue(strutils.bool_from_string(c('true')))
        self.assertTrue(strutils.bool_from_string(c('TRUE')))
        self.assertTrue(strutils.bool_from_string(c('on')))
        self.assertTrue(strutils.bool_from_string(c('On')))
        self.assertTrue(strutils.bool_from_string(c('yes')))
        self.assertTrue(strutils.bool_from_string(c('YES')))
        self.assertTrue(strutils.bool_from_string(c('yEs')))
        self.assertTrue(strutils.bool_from_string(c('1')))
        self.assertTrue(strutils.bool_from_string(c('T')))
        self.assertTrue(strutils.bool_from_string(c('t')))
        self.assertTrue(strutils.bool_from_string(c('Y')))
        self.assertTrue(strutils.bool_from_string(c('y')))
        self.assertFalse(strutils.bool_from_string(c('false')))
        self.assertFalse(strutils.bool_from_string(c('FALSE')))
        self.assertFalse(strutils.bool_from_string(c('off')))
        self.assertFalse(strutils.bool_from_string(c('OFF')))
        self.assertFalse(strutils.bool_from_string(c('no')))
        self.assertFalse(strutils.bool_from_string(c('0')))
        self.assertFalse(strutils.bool_from_string(c('42')))
        self.assertFalse(strutils.bool_from_string(c('This should not be True')))
        self.assertFalse(strutils.bool_from_string(c('F')))
        self.assertFalse(strutils.bool_from_string(c('f')))
        self.assertFalse(strutils.bool_from_string(c('N')))
        self.assertFalse(strutils.bool_from_string(c('n')))
        self.assertTrue(strutils.bool_from_string(c(' 1 ')))
        self.assertTrue(strutils.bool_from_string(c(' true ')))
        self.assertFalse(strutils.bool_from_string(c(' 0 ')))
        self.assertFalse(strutils.bool_from_string(c(' false ')))

    def test_bool_from_string(self):
        self._test_bool_from_string(lambda s: s)

    def test_unicode_bool_from_string(self):
        self._test_bool_from_string(str)
        self.assertFalse(strutils.bool_from_string('使用', strict=False))
        exc = self.assertRaises(ValueError, strutils.bool_from_string, '使用', strict=True)
        expected_msg = "Unrecognized value '使用', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
        self.assertEqual(expected_msg, str(exc))

    def test_other_bool_from_string(self):
        self.assertFalse(strutils.bool_from_string(None))
        self.assertFalse(strutils.bool_from_string(mock.Mock()))

    def test_int_bool_from_string(self):
        self.assertTrue(strutils.bool_from_string(1))
        self.assertFalse(strutils.bool_from_string(-1))
        self.assertFalse(strutils.bool_from_string(0))
        self.assertFalse(strutils.bool_from_string(2))

    def test_strict_bool_from_string(self):
        exc = self.assertRaises(ValueError, strutils.bool_from_string, None, strict=True)
        expected_msg = "Unrecognized value 'None', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
        self.assertEqual(expected_msg, str(exc))
        self.assertFalse(strutils.bool_from_string('Other', strict=False))
        exc = self.assertRaises(ValueError, strutils.bool_from_string, 'Other', strict=True)
        expected_msg = "Unrecognized value 'Other', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
        self.assertEqual(expected_msg, str(exc))
        exc = self.assertRaises(ValueError, strutils.bool_from_string, 2, strict=True)
        expected_msg = "Unrecognized value '2', acceptable values are: '0', '1', 'f', 'false', 'n', 'no', 'off', 'on', 't', 'true', 'y', 'yes'"
        self.assertEqual(expected_msg, str(exc))
        self.assertFalse(strutils.bool_from_string('f', strict=True))
        self.assertFalse(strutils.bool_from_string('false', strict=True))
        self.assertFalse(strutils.bool_from_string('off', strict=True))
        self.assertFalse(strutils.bool_from_string('n', strict=True))
        self.assertFalse(strutils.bool_from_string('no', strict=True))
        self.assertFalse(strutils.bool_from_string('0', strict=True))
        self.assertTrue(strutils.bool_from_string('1', strict=True))
        for char in ('O', 'o', 'L', 'l', 'I', 'i'):
            self.assertRaises(ValueError, strutils.bool_from_string, char, strict=True)

    def test_int_from_bool_as_string(self):
        self.assertEqual(1, strutils.int_from_bool_as_string(True))
        self.assertEqual(0, strutils.int_from_bool_as_string(False))

    def test_is_valid_boolstr(self):
        self.assertTrue(strutils.is_valid_boolstr('true'))
        self.assertTrue(strutils.is_valid_boolstr('false'))
        self.assertTrue(strutils.is_valid_boolstr('yes'))
        self.assertTrue(strutils.is_valid_boolstr('no'))
        self.assertTrue(strutils.is_valid_boolstr('y'))
        self.assertTrue(strutils.is_valid_boolstr('n'))
        self.assertTrue(strutils.is_valid_boolstr('1'))
        self.assertTrue(strutils.is_valid_boolstr('0'))
        self.assertTrue(strutils.is_valid_boolstr(1))
        self.assertTrue(strutils.is_valid_boolstr(0))
        self.assertFalse(strutils.is_valid_boolstr('maybe'))
        self.assertFalse(strutils.is_valid_boolstr('only on tuesdays'))

    def test_slugify(self):
        to_slug = strutils.to_slug
        self.assertRaises(TypeError, to_slug, True)
        self.assertEqual('hello', to_slug('hello'))
        self.assertEqual('two-words', to_slug('Two Words'))
        self.assertEqual('ma-any-spa-ce-es', to_slug('Ma-any\t spa--ce- es'))
        self.assertEqual('excamation', to_slug('exc!amation!'))
        self.assertEqual('ampserand', to_slug('&ampser$and'))
        self.assertEqual('ju5tnum8er', to_slug('ju5tnum8er'))
        self.assertEqual('strip-', to_slug(' strip - '))
        self.assertEqual('perche', to_slug('perchÃ©'.encode('latin-1')))
        self.assertEqual('strange', to_slug('\x80strange', errors='ignore'))