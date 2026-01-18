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
def _RunInApp(function, args, kwargs):
    """Executes a set of Python unit tests, ensuring app.really_start.

  Most users should call basetest.main() instead of _RunInApp.

  _RunInApp calculates argv to be the command-line arguments of this program
  (without the flags), sets the default of FLAGS.alsologtostderr to True,
  then it calls function(argv, args, kwargs), making sure that `function'
  will get called within app.run() or app.really_start(). _RunInApp does this
  by checking whether it is called by either app.run() or
  app.really_start(), or by calling app.really_start() explicitly.

  The reason why app.really_start has to be ensured is to make sure that
  flags are parsed and stripped properly, and other initializations done by
  the app module are also carried out, no matter if basetest.run() is called
  from within or outside app.run().

  If _RunInApp is called from within app.run(), then it will reparse
  sys.argv and pass the result without command-line flags into the argv
  argument of `function'. The reason why this parsing is needed is that
  __main__.main() calls basetest.main() without passing its argv. So the
  only way _RunInApp could get to know the argv without the flags is that
  it reparses sys.argv.

  _RunInApp changes the default of FLAGS.alsologtostderr to True so that the
  test program's stderr will contain all the log messages unless otherwise
  specified on the command-line. This overrides any explicit assignment to
  FLAGS.alsologtostderr by the test program prior to the call to _RunInApp()
  (e.g. in __main__.main).

  Please note that _RunInApp (and the function it calls) is allowed to make
  changes to kwargs.

  Args:
    function: basetest.RunTests or a similar function. It will be called as
      function(argv, args, kwargs) where argv is a list containing the
      elements of sys.argv without the command-line flags.
    args: Positional arguments passed through to unittest.TestProgram.__init__.
    kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
    if _IsInAppMain():
        saved_flags = dict(((f.name, SavedFlag(f)) for f in FLAGS.FlagDict().itervalues()))
        if 'alsologtostderr' in saved_flags:
            FLAGS.alsologtostderr = True
            del saved_flags['alsologtostderr']
        argv = FLAGS(sys.argv)
        for saved_flag in saved_flags.itervalues():
            saved_flag.RestoreFlag()
        function(argv, args, kwargs)
    else:
        if 'alsologtostderr' in FLAGS:
            FLAGS.SetDefault('alsologtostderr', True)

        def Main(argv):
            function(argv, args, kwargs)
        app.really_start(main=Main)