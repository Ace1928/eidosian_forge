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
def GetProperties(self):
    """Gets the properties defined in this configuration.

    These are the properties literally defined in this file, not the effective
    properties based on other configs or the environment.

    Returns:
      {str, str}, A dictionary of all properties defined in this configuration
      file.
    """
    if not self.file_path:
        return {}
    return properties_file.PropertiesFile([self.file_path]).AllProperties()