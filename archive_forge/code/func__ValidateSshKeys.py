from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _ValidateSshKeys(metadata_dict):
    """Validates the ssh-key entries in metadata.

  The ssh-key entry in metadata should start with <username> and it cannot
  be a private key
  (i.e. <username>:ssh-rsa <key-blob> <username>@<example.com> or
  <username>:ssh-rsa <key-blob>
  google-ssh {"userName": <username>@<example.com>, "expireOn": <date>}
  when the key can expire.)

  Args:
    metadata_dict: A dictionary object containing metadata.

  Raises:
    InvalidSshKeyException: If the <username> at the front is missing or private
    key(s) are detected.
  """
    ssh_keys = metadata_dict.get(constants.SSH_KEYS_METADATA_KEY, '')
    ssh_keys_legacy = metadata_dict.get(constants.SSH_KEYS_LEGACY_METADATA_KEY, '')
    ssh_keys_combined = '\n'.join((ssh_keys, ssh_keys_legacy))
    if 'PRIVATE KEY' in ssh_keys_combined:
        raise InvalidSshKeyException('Private key(s) are detected. Note that only public keys should be added.')
    keys = ssh_keys_combined.split('\n')
    keys_missing_username = []
    for key in keys:
        if key and _SshKeyStartsWithKeyType(key):
            keys_missing_username.append(key)
    if keys_missing_username:
        message = 'The following key(s) are missing the <username> at the front\n{}\n\nFormat ssh keys following https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys'
        message_content = message.format('\n'.join(keys_missing_username))
        raise InvalidSshKeyException(message_content)