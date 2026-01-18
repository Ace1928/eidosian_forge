from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
def _EffectiveActiveConfigName():
    """Gets the currently active configuration.

  It checks (in order):
    - Flag values
    - Environment variable values
    - The value set in the activator file

  Returns:
    str, The name of the active configuration or None if no location declares
    an active configuration.
  """
    config_name = FLAG_OVERRIDE_STACK.ActiveConfig()
    if not config_name:
        config_name = _ActiveConfigNameFromEnv()
    if not config_name:
        config_name = _ActiveConfigNameFromFile()
    return config_name