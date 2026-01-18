import sys
import types
from testtools import TestCase
from testtools.matchers import (
from extras import (
class TestTryImports(TestCase):

    def test_doesnt_exist(self):
        marker = object()
        result = try_imports(['doesntexist'], marker)
        self.assertThat(result, Is(marker))

    def test_fallback(self):
        result = try_imports(['doesntexist', 'os'])
        import os
        self.assertThat(result, Is(os))

    def test_None_is_default_alternative(self):
        e = self.assertRaises(ImportError, try_imports, ['doesntexist', 'noreally'])
        self.assertThat(str(e), Equals('Could not import any of: doesntexist, noreally'))

    def test_existing_module(self):
        result = try_imports(['os'], object())
        import os
        self.assertThat(result, Is(os))

    def test_existing_submodule(self):
        result = try_imports(['os.path'], object())
        import os
        self.assertThat(result, Is(os.path))

    def test_nonexistent_submodule(self):
        marker = object()
        result = try_imports(['os.doesntexist'], marker)
        self.assertThat(result, Is(marker))

    def test_fallback_submodule(self):
        result = try_imports(['os.doesntexist', 'os.path'])
        import os
        self.assertThat(result, Is(os.path))

    def test_error_callback(self):
        check_error_callback(self, try_imports, ['os.doesntexist', 'os.notthiseither'], 2, False)
        check_error_callback(self, try_imports, ['os.doesntexist', 'os.notthiseither', 'os'], 2, True)
        check_error_callback(self, try_imports, ['os.path'], 0, True)