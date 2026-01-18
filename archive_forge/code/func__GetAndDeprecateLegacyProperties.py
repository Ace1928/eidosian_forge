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
def _GetAndDeprecateLegacyProperties(paths):
    """Gets the contents of the legacy  properties to include in a new config.

  If the properties have already been imported, this returns nothing.  If not,
  this will return the old properties and mark the old file as deprecated.

  Args:
    paths: config.Paths, The instantiated Paths object to use to calculate the
      location.

  Returns:
    str, The contents of the legacy properties file or ''.
  """
    contents = ''
    if os.path.exists(paths.user_properties_path):
        contents = file_utils.ReadFileContents(paths.user_properties_path)
        if contents.startswith(_LEGACY_DEPRECATION_MESSAGE):
            contents = ''
        else:
            file_utils.WriteFileContents(paths.user_properties_path, _LEGACY_DEPRECATION_MESSAGE + contents)
    return contents