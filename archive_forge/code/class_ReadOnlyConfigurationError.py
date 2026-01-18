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
class ReadOnlyConfigurationError(Error):
    """An exception for when the active config is read-only (e.g. None)."""

    def __init__(self, config_name):
        super(ReadOnlyConfigurationError, self).__init__('Properties in configuration [{0}] cannot be set.'.format(config_name))