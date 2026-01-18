from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shlex
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
import six
def MatchEnvVars(word, env_vars):
    """Returns environment variables beginning with `word`.

  Args:
    word: The word that is being compared to environment variables.
    env_vars: The list of environment variables.

  Returns:
    []: No completions.
    [completions]: List, all possible sorted completions.
  """
    completions = []
    prefix = word[1:]
    for child in env_vars:
        if child.startswith(prefix):
            if platforms.OperatingSystem.IsWindows():
                completions.append('%' + child + '%')
            else:
                completions.append('$' + child)
    return completions