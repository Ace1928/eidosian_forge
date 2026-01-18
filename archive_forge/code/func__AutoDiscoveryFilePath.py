from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import enum
import json
import os
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import _mtls_helper
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _AutoDiscoveryFilePath():
    """Return the file path of the context aware configuration file."""
    cfg_file = properties.VALUES.context_aware.auto_discovery_file_path.Get()
    if cfg_file is not None:
        return cfg_file
    return DEFAULT_AUTO_DISCOVERY_FILE_PATH