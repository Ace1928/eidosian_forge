from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.command_lib.iot import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import times
import six
def ParseCredentials(public_keys, messages=None):
    """Parse a DeviceCredential from user-supplied arguments.

  Returns a list of DeviceCredential with the appropriate type, expiration time
  (if provided), and contents of the file for each public key.

  Args:
    public_keys: list of dict (maximum 3) representing public key credentials.
      The dict should have the following keys:
      - 'type': Required. The key type. One of [es256, rs256]
      - 'path': Required. Path to a valid key file on disk.
      - 'expiration-time': Optional. datetime, the expiration time for the
        credential.
    messages: module or None, the apitools messages module for Cloud IoT (uses a
      default module if not provided).

  Returns:
    List of DeviceCredential (possibly empty).

  Raises:
    TypeError: if an invalid public_key specification is given in public_keys
    ValueError: if an invalid public key type is given (that is, neither es256
      nor rs256)
    InvalidPublicKeySpecificationError: if a public_key specification is missing
      a required part, or too many public keys are provided.
    InvalidKeyFileError: if a valid combination of flags is given, but the
      specified key file is not valid or not readable.
  """
    messages = messages or devices.GetMessagesModule()
    if not public_keys:
        return []
    if len(public_keys) > MAX_PUBLIC_KEY_NUM:
        raise InvalidPublicKeySpecificationError('Too many public keys specified: [{}] given, but maximum [{}] allowed.'.format(len(public_keys), MAX_PUBLIC_KEY_NUM))
    credentials = []
    for key in public_keys:
        _ValidatePublicKeyDict(key)
        credentials.append(ParseCredential(key.get('path'), key.get('type'), key.get('expiration-time'), messages=messages))
    return credentials