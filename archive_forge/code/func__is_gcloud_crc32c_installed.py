from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import struct
import textwrap
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _is_gcloud_crc32c_installed():
    """Returns if gcloud-crc32c is installed and does not attempt install."""
    try:
        return BINARY_NAME in binary_operations.CheckForInstalledBinary(BINARY_NAME)
    except binary_operations.MissingExecutableException:
        return False