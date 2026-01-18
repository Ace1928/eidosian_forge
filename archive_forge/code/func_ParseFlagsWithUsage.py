from mx import DateTime
import os
import pdb
import sys
import traceback
from google.apputils import app
import gflags as flags
def ParseFlagsWithUsage(argv):
    """Parse the flags, exiting (after printing usage) if they are unparseable.

  Args:
    argv: command line arguments

  Returns:
    remaining command line arguments after parsing flags
  """
    global _cmd_argv
    try:
        _cmd_argv = FLAGS(argv)
        return _cmd_argv
    except flags.FlagsError as error:
        ShortHelpAndExit('FATAL Flags parsing error: %s' % error)