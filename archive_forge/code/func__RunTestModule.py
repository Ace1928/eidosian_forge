from distutils import errors
import imp
import os
import re
import shlex
import sys
import traceback
from setuptools.command import test
def _RunTestModule(self, module_path):
    """Run a module as a test module given its path.

    Args:
      module_path: The path to the module to test; must end in '.py'.

    Returns:
      True if the tests in this module pass, False if not or if an error occurs.
    """
    path, filename = os.path.split(module_path)
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    old_modules = sys.modules.copy()
    sys.path.insert(0, path)
    module_name = filename.replace('.py', '')
    import_tuple = imp.find_module(module_name, [path])
    module = imp.load_module(module_name, *import_tuple)
    sys.modules['__main__'] = module
    sys.argv = [module.__file__] + self.test_args
    import basetest
    try:
        try:
            sys.stderr.write('Testing %s\n' % module_name)
            basetest.main()
            return False
        except SystemExit as e:
            returncode, = e.args
            return not returncode
        except:
            traceback.print_exc()
            return False
    finally:
        sys.argv[:] = old_argv
        sys.path[:] = old_path
        sys.modules.clear()
        sys.modules.update(old_modules)