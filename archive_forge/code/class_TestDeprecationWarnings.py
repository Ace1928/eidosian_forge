import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
class TestDeprecationWarnings(TestCase):

    def capture_warning(self, message, category, stacklevel=None):
        self._warnings.append((message, category, stacklevel))

    def setUp(self):
        super().setUp()
        self._warnings = []

    @deprecated_method(deprecated_in((0, 7, 0)))
    def deprecated_method(self):
        """Deprecated method docstring.

        This might explain stuff.
        """
        return 1

    @staticmethod
    @deprecated_function(deprecated_in((0, 7, 0)))
    def deprecated_static():
        """Deprecated static."""
        return 1

    def test_deprecated_static(self):
        expected_warning = ('breezy.tests.test_symbol_versioning.TestDeprecationWarnings.deprecated_static was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated static.\n\nThis function was deprecated in version 0.7.0.\n'
        self.check_deprecated_callable(expected_warning, expected_docstring, 'deprecated_static', 'breezy.tests.test_symbol_versioning', self.deprecated_static)

    def test_deprecated_method(self):
        expected_warning = ('breezy.tests.test_symbol_versioning.TestDeprecationWarnings.deprecated_method was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated method docstring.\n\n        This might explain stuff.\n        \n        This method was deprecated in version 0.7.0.\n        '
        self.check_deprecated_callable(expected_warning, expected_docstring, 'deprecated_method', 'breezy.tests.test_symbol_versioning', self.deprecated_method)

    def test_deprecated_function(self):
        expected_warning = ('breezy.tests.test_symbol_versioning.sample_deprecated_function was deprecated in version 0.7.0.', DeprecationWarning, 2)
        expected_docstring = 'Deprecated function docstring.\n\nThis function was deprecated in version 0.7.0.\n'
        self.check_deprecated_callable(expected_warning, expected_docstring, 'sample_deprecated_function', 'breezy.tests.test_symbol_versioning', sample_deprecated_function)

    def test_deprecated_list(self):
        expected_warning = ("Modifying a_deprecated_list was deprecated in version 0.9.0. Don't use me", DeprecationWarning, 3)
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(['one'], a_deprecated_list)
            self.assertEqual([], self._warnings)
            a_deprecated_list.append('foo')
            self.assertEqual([expected_warning], self._warnings)
            self.assertEqual(['one', 'foo'], a_deprecated_list)
            a_deprecated_list.extend(['bar', 'baz'])
            self.assertEqual([expected_warning] * 2, self._warnings)
            self.assertEqual(['one', 'foo', 'bar', 'baz'], a_deprecated_list)
            a_deprecated_list.insert(1, 'xxx')
            self.assertEqual([expected_warning] * 3, self._warnings)
            self.assertEqual(['one', 'xxx', 'foo', 'bar', 'baz'], a_deprecated_list)
            a_deprecated_list.remove('foo')
            self.assertEqual([expected_warning] * 4, self._warnings)
            self.assertEqual(['one', 'xxx', 'bar', 'baz'], a_deprecated_list)
            val = a_deprecated_list.pop()
            self.assertEqual([expected_warning] * 5, self._warnings)
            self.assertEqual('baz', val)
            self.assertEqual(['one', 'xxx', 'bar'], a_deprecated_list)
            val = a_deprecated_list.pop(1)
            self.assertEqual([expected_warning] * 6, self._warnings)
            self.assertEqual('xxx', val)
            self.assertEqual(['one', 'bar'], a_deprecated_list)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def test_deprecated_dict(self):
        expected_warning = ('access to a_deprecated_dict was deprecated in version 0.14.0. Pull the other one!', DeprecationWarning, 2)
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(len(a_deprecated_dict), 1)
            self.assertEqual([expected_warning], self._warnings)
            a_deprecated_dict['b'] = 42
            self.assertEqual(a_deprecated_dict['b'], 42)
            self.assertTrue('b' in a_deprecated_dict)
            del a_deprecated_dict['b']
            self.assertFalse('b' in a_deprecated_dict)
            self.assertEqual([expected_warning] * 6, self._warnings)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def check_deprecated_callable(self, expected_warning, expected_docstring, expected_name, expected_module, deprecated_callable):
        if __doc__ is None:
            expected_docstring = expected_docstring.split('\n')[-2].lstrip()
        old_warning_method = symbol_versioning.warn
        try:
            symbol_versioning.set_warning_method(self.capture_warning)
            self.assertEqual(1, deprecated_callable())
            self.assertEqual([expected_warning], self._warnings)
            deprecated_callable()
            self.assertEqual([expected_warning, expected_warning], self._warnings)
            self.assertEqualDiff(expected_docstring, deprecated_callable.__doc__)
            self.assertEqualDiff(expected_name, deprecated_callable.__name__)
            self.assertEqualDiff(expected_module, deprecated_callable.__module__)
            self.assertTrue(deprecated_callable.is_deprecated)
        finally:
            symbol_versioning.set_warning_method(old_warning_method)

    def test_deprecated_passed(self):
        self.assertEqual(True, symbol_versioning.deprecated_passed(None))
        self.assertEqual(True, symbol_versioning.deprecated_passed(True))
        self.assertEqual(True, symbol_versioning.deprecated_passed(False))
        self.assertEqual(False, symbol_versioning.deprecated_passed(symbol_versioning.DEPRECATED_PARAMETER))

    def test_deprecation_string(self):
        """We can get a deprecation string for a method or function."""
        err_message = symbol_versioning.deprecation_string(self.test_deprecation_string, deprecated_in((0, 11, 0)))
        self.assertEqual(err_message, 'breezy.tests.test_symbol_versioning.TestDeprecationWarnings.test_deprecation_string was deprecated in version 0.11.0.')
        self.assertEqual('breezy.symbol_versioning.deprecated_function was deprecated in version 0.11.0.', symbol_versioning.deprecation_string(symbol_versioning.deprecated_function, deprecated_in((0, 11, 0))))