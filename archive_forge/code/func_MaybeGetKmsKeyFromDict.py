from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def MaybeGetKmsKeyFromDict(args, messages, current_value, conflicting_arg='--csek-key-file'):
    """Gets the Cloud KMS CryptoKey reference for a boot disk's initialize params.

  Args:
    args: A dictionary of a boot disk's initialize params.
    messages: Compute API messages module.
    current_value: Current CustomerEncryptionKey value.
    conflicting_arg: name of conflicting argument

  Returns:
    CustomerEncryptionKey message with the KMS key populated if args has a key.
  Raises:
    ConflictingArgumentsException if an encryption key is already populated.
  """
    if bool(_GetSpecifiedKmsDict(args)):
        if current_value:
            raise calliope_exceptions.ConflictingArgumentsException(conflicting_arg, *_GetSpecifiedKmsArgs(args))
        return _DictToMessage(args, messages)
    return current_value