from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def ShortHelpAndExit(message=None):
    """Display optional message, followed by a note on how to get help, then exit.

  Args:
    message: optional message to display
  """
    sys.stdout.flush()
    if message is not None:
        sys.stderr.write('%s\n' % message)
    sys.stderr.write("Run '%s help' to get help\n" % GetAppBasename())
    sys.exit(1)