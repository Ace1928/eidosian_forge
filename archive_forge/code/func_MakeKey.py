from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
@staticmethod
def MakeKey(key_material, key_type, allow_rsa_encrypted=False):
    """Make a CSEK key.

    Args:
      key_material: str, the key material for this key
      key_type: str, the type of this key
      allow_rsa_encrypted: bool, whether the key is allowed to be RSA-wrapped

    Returns:
      CsekRawKey or CsekRsaEncryptedKey derived from the given key material and
      type.

    Raises:
      BadKeyTypeException: if the key is not a valid key type
    """
    if key_type == 'raw':
        return CsekRawKey(key_material)
    if key_type == 'rsa-encrypted':
        if allow_rsa_encrypted:
            return CsekRsaEncryptedKey(key_material)
        raise BadKeyTypeException(key_type, 'this feature is only allowed in the alpha and beta versions of this command.')
    raise BadKeyTypeException(key_type)