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
def _EnsureValidConfigName(config_name, allow_reserved):
    """Ensures that the given configuration name conforms to the standard.

  Args:
    config_name: str, The name to check.
    allow_reserved: bool, Allows the given name to be one of the reserved
      configuration names.

  Raises:
    InvalidConfigName: If the name is invalid.
  """
    if not _IsValidConfigName(config_name, allow_reserved):
        raise InvalidConfigName(config_name)