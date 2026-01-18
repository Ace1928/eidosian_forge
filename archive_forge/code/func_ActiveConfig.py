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
@staticmethod
def ActiveConfig():
    """Gets the currently active configuration.

    There must always be an active configuration.  If there isn't this means
    no configurations have been created yet and this will auto-create a default
    configuration.  If there are legacy user properties, they will be migrated
    to the newly created configuration.

    Returns:
      Configuration, the currently active configuration.
    """
    return ActiveConfig(force_create=True)