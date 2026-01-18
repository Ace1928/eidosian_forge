from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import collections
import datetime
import json
import os
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
from googlecloudsdk.core.util.files import FileReader
from googlecloudsdk.core.util.files import FileWriter
import six
def _PrepareSSHKeysValue(ssh_keys):
    """Returns a string appropriate for the metadata.

  Expired SSH keys are always removed.
  Then Values are taken from the tail until either all values are taken or
  _MAX_METADATA_VALUE_SIZE_IN_BYTES is reached, whichever comes first. The
  selected values are then reversed. Only values at the head of the list will be
  subject to removal.

  Args:
    ssh_keys: A list of keys. Each entry should be one key.

  Returns:
    A new-line-joined string of SSH keys.
  """
    keys = []
    bytes_consumed = 0
    now = times.LocalizeDateTime(times.Now(), times.UTC)
    for key in reversed(ssh_keys):
        try:
            expiration = _SSHKeyExpiration(key)
            expired = expiration is not None and expiration < now
            if expired:
                continue
        except (ValueError, times.DateTimeSyntaxError, times.DateTimeValueError) as exc:
            log.warning('Treating {0!r} as unexpiring, since unable to parse: {1}'.format(key, exc))
        num_bytes = len(key + '\n')
        if bytes_consumed + num_bytes > constants.MAX_METADATA_VALUE_SIZE_IN_BYTES:
            prompt_message = 'The following SSH key will be removed from your project because your SSH keys metadata value has reached its maximum allowed size of {0} bytes: {1}'
            prompt_message = prompt_message.format(constants.MAX_METADATA_VALUE_SIZE_IN_BYTES, key)
            console_io.PromptContinue(message=prompt_message, cancel_on_no=True)
        else:
            keys.append(key)
            bytes_consumed += num_bytes
    keys.reverse()
    return '\n'.join(keys)