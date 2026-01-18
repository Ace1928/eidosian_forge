from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import sys
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _GetAndUpdateRcPath(completion_update, path_update, rc_path, host_os):
    """Returns an rc path based on the default rc path or user input.

  Gets default rc path based on environment. If prompts are enabled,
  allows user to update to preferred file path. Otherwise, prints a warning
  that the default rc path will be updated.

  Args:
    completion_update: bool, Whether or not to do command completion.
    path_update: bool, Whether or not to update PATH.
    rc_path: str, the rc path given by the user, from --rc-path arg.
    host_os: str, The host os identification string.

  Returns:
    str, A path to the rc file to update.
  """
    if not (completion_update or path_update):
        return None
    if rc_path:
        return rc_path
    preferred_shell = _GetPreferredShell(encoding.GetEncodedValue(os.environ, 'SHELL', '/bin/sh'))
    default_rc_path = os.path.join(files.GetHomeDir(), _GetShellRcFileName(preferred_shell, host_os))
    if not console_io.CanPrompt():
        _TraceAction('You specified that you wanted to update your rc file. The default file will be updated: [{rc_path}]'.format(rc_path=default_rc_path))
        return default_rc_path
    rc_path_update = console_io.PromptResponse('The Google Cloud SDK installer will now prompt you to update an rc file to bring the Google Cloud CLIs into your environment.\n\nEnter a path to an rc file to update, or leave blank to use [{rc_path}]:  '.format(rc_path=default_rc_path))
    return files.ExpandHomeDir(rc_path_update) if rc_path_update else default_rc_path