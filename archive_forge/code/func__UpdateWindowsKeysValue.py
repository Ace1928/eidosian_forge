from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import openssl_encryption_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util import gaia
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as core_encoding
from googlecloudsdk.core.util import files
def _UpdateWindowsKeysValue(self, existing_metadata):
    """Returns a string appropriate for the metadata.

    Values are removed if they have expired and non-expired keys are removed
    from the head of the list only if the total key size is greater than
    MAX_METADATA_VALUE_SIZE_IN_BYTES.

    Args:
      existing_metadata: The existing metadata for the instance to be updated.

    Returns:
      A new-line-joined string of Windows keys.
    """
    windows_keys = []
    self.old_metadata_keys = []
    for item in existing_metadata.items:
        if item.key == METADATA_KEY:
            windows_keys = [key.strip() for key in item.value.split('\n') if key]
        if item.key in OLD_METADATA_KEYS:
            self.old_metadata_keys.append(item.key)
    windows_keys.append(self.windows_key_entry)
    keys = []
    bytes_consumed = 0
    for key in reversed(windows_keys):
        num_bytes = len(key + '\n')
        key_expired = False
        try:
            key_data = json.loads(key)
            if time_util.IsExpired(key_data['expireOn']):
                key_expired = True
        except (ValueError, KeyError):
            pass
        if key_expired:
            log.debug('The following Windows key has expired and will be removed from your project: {0}'.format(key))
        elif bytes_consumed + num_bytes > constants.MAX_METADATA_VALUE_SIZE_IN_BYTES:
            log.debug('The following Windows key will be removed from your project because your windows keys metadata value has reached its maximum allowed size of {0} bytes: {1}'.format(constants.MAX_METADATA_VALUE_SIZE_IN_BYTES, key))
        else:
            keys.append(key)
            bytes_consumed += num_bytes
    log.debug('Number of Windows Keys: {0}'.format(len(keys)))
    keys.reverse()
    return '\n'.join(keys)