import os
import re
import sys
import traceback
import types
import functools
import warnings
from fnmatch import fnmatch, fnmatchcase
from . import case, suite, util
def _find_test_path(self, full_path, pattern):
    """Used by discovery.

        Loads tests from a single file, or a directories' __init__.py when
        passed the directory.

        Returns a tuple (None_or_tests_from_file, should_recurse).
        """
    basename = os.path.basename(full_path)
    if os.path.isfile(full_path):
        if not VALID_MODULE_NAME.match(basename):
            return (None, False)
        if not self._match_path(basename, full_path, pattern):
            return (None, False)
        name = self._get_name_from_path(full_path)
        try:
            module = self._get_module_from_name(name)
        except case.SkipTest as e:
            return (_make_skipped_test(name, e, self.suiteClass), False)
        except:
            error_case, error_message = _make_failed_import_test(name, self.suiteClass)
            self.errors.append(error_message)
            return (error_case, False)
        else:
            mod_file = os.path.abspath(getattr(module, '__file__', full_path))
            realpath = _jython_aware_splitext(os.path.realpath(mod_file))
            fullpath_noext = _jython_aware_splitext(os.path.realpath(full_path))
            if realpath.lower() != fullpath_noext.lower():
                module_dir = os.path.dirname(realpath)
                mod_name = _jython_aware_splitext(os.path.basename(full_path))
                expected_dir = os.path.dirname(full_path)
                msg = '%r module incorrectly imported from %r. Expected %r. Is this module globally installed?'
                raise ImportError(msg % (mod_name, module_dir, expected_dir))
            return (self.loadTestsFromModule(module, pattern=pattern), False)
    elif os.path.isdir(full_path):
        if not os.path.isfile(os.path.join(full_path, '__init__.py')):
            return (None, False)
        load_tests = None
        tests = None
        name = self._get_name_from_path(full_path)
        try:
            package = self._get_module_from_name(name)
        except case.SkipTest as e:
            return (_make_skipped_test(name, e, self.suiteClass), False)
        except:
            error_case, error_message = _make_failed_import_test(name, self.suiteClass)
            self.errors.append(error_message)
            return (error_case, False)
        else:
            load_tests = getattr(package, 'load_tests', None)
            self._loading_packages.add(name)
            try:
                tests = self.loadTestsFromModule(package, pattern=pattern)
                if load_tests is not None:
                    return (tests, False)
                return (tests, True)
            finally:
                self._loading_packages.discard(name)
    else:
        return (None, False)