from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def WarnAndExitOnBlockedCommand(args, blocked_commands):
    """Block certain subcommands, warn the user, and exit.

  Args:
    args: the command line arguments, including the 0th argument which is
      the program name.
    blocked_commands: a map of blocked commands to the messages that should be
      printed when they're run.
  """
    bad_arg = None
    for arg in args[1:]:
        if arg and arg[0] == '-':
            continue
        if arg in blocked_commands:
            bad_arg = arg
            break
    blocked = bad_arg is not None
    if blocked:
        sys.stderr.write('It looks like you are trying to run "%s %s".\n' % (args[0], bad_arg))
        sys.stderr.write('The "%s" command is no longer needed with Google Cloud CLI.\n' % bad_arg)
        sys.stderr.write(blocked_commands[bad_arg] + '\n')
        answer = input('Really run this command? (y/N) ')
        if answer not in ['y', 'Y']:
            sys.exit(1)