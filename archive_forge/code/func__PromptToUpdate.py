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
def _PromptToUpdate(path_update, completion_update):
    """Prompt the user to update path or command completion if unspecified.

  Args:
    path_update: bool, Value of the --update-path arg.
    completion_update: bool, Value of the --command-completion arg.

  Returns:
    (path_update, completion_update) (bool, bool) Whether to update path and
        enable completion, respectively, after prompting the user.
  """
    if path_update is not None and completion_update is not None:
        return (path_update, completion_update)
    actions = []
    if path_update is None:
        actions.append(_PATH_PROMPT)
    if completion_update is None:
        actions.append(_COMPLETION_PROMPT)
    prompt = '\nModify profile to {}?'.format(' and '.join(actions))
    response = console_io.PromptContinue(prompt)
    path_update = response if path_update is None else path_update
    completion_update = response if completion_update is None else completion_update
    return (path_update, completion_update)