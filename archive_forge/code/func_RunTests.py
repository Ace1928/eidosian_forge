import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def RunTests(argv, args, kwargs):
    """Executes a set of Python unit tests within app.really_start.

  Most users should call basetest.main() instead of RunTests.

  Please note that RunTests should be called from app.really_start (which is
  called from app.run()). Calling basetest.main() would ensure that.

  Please note that RunTests is allowed to make changes to kwargs.

  Args:
    argv: sys.argv with the command-line flags removed from the front, i.e. the
      argv with which app.run() has called __main__.main.
    args: Positional arguments passed through to unittest.TestProgram.__init__.
    kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
    test_runner = kwargs.get('testRunner')
    if not os.path.isdir(FLAGS.test_tmpdir):
        os.makedirs(FLAGS.test_tmpdir)
    main_mod = sys.modules['__main__']
    if hasattr(main_mod, 'setUp') and callable(main_mod.setUp):
        main_mod.setUp()
    kwargs.setdefault('argv', argv)
    try:
        result = None
        test_program = TestProgramManualRun(*args, **kwargs)
        if test_runner:
            test_program.testRunner = test_runner
        else:
            test_program.testRunner = unittest.TextTestRunner(verbosity=test_program.verbosity)
        result = test_program.testRunner.run(test_program.test)
    finally:
        if hasattr(main_mod, 'tearDown') and callable(main_mod.tearDown):
            main_mod.tearDown()
    sys.exit(not result.wasSuccessful())