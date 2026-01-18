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
def _ActiveConfigNameFromEnv():
    """Gets the currently active configuration according to the environment.

  Returns:
    str, The name of the active configuration or None.
  """
    return encoding.GetEncodedValue(os.environ, config.CLOUDSDK_ACTIVE_CONFIG_NAME, None)