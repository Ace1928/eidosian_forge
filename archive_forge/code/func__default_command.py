from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
from boto import config
import gslib
from gslib import exception
from gslib.utils import boto_util
from gslib.utils import execution_util
def _default_command():
    """Loads default cert provider command.

  Returns:
      str: The default command.

  Raises:
      CertProvisionError: If command cannot be found.
  """
    metadata_path = _check_path()
    if not metadata_path:
        raise CertProvisionError('Client certificate provider file not found.')
    metadata_json = _read_metadata_file(metadata_path)
    if _CERT_PROVIDER_COMMAND not in metadata_json:
        raise CertProvisionError('Client certificate provider command not found.')
    command = metadata_json[_CERT_PROVIDER_COMMAND]
    if _CERT_PROVIDER_COMMAND_PASSPHRASE_OPTION not in command:
        command.append(_CERT_PROVIDER_COMMAND_PASSPHRASE_OPTION)
    return command