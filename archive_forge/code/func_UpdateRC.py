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
def UpdateRC(completion_update, path_update, rc_path, bin_path, sdk_root):
    """Update the system path to include bin_path.

  Args:
    completion_update: bool, Whether or not to do command completion. From
      --command-completion arg during install. If None, ask.
    path_update: bool, Whether or not to update PATH. From --path-update arg
      during install. If None, ask.
    rc_path: str, The path to the rc file to update. From --rc-path during
      install. If None, ask.
    bin_path: str, The absolute path to the directory that will contain
      Cloud SDK binaries.
    sdk_root: str, The path to the Cloud SDK root.
  """
    host_os = platforms.OperatingSystem.Current()
    if host_os == platforms.OperatingSystem.WINDOWS:
        if path_update is None:
            path_update = console_io.PromptContinue(prompt_string='Update %PATH% to include Cloud SDK binaries?')
        if path_update:
            _UpdatePathForWindows(bin_path)
        return
    if console_io.CanPrompt():
        path_update, completion_update = _PromptToUpdate(path_update, completion_update)
    elif rc_path and (path_update is None and completion_update is None):
        path_update = True
        completion_update = True
        _TraceAction('Profile will be modified to {} and {}.'.format(_PATH_PROMPT, _COMPLETION_PROMPT))
    _GetRcUpdater(completion_update, path_update, rc_path, sdk_root, host_os).Update()