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
def RemoveEntries(message_classes, existing_metadata, keys=None, remove_all=False):
    """Removes keys from existing_metadata.

  Args:
    message_classes: An object containing API message classes.
    existing_metadata: The Metadata message to remove keys from.
    keys: The keys to remove. This can be None if remove_all is True.
    remove_all: If True, all entries from existing_metadata are
      removed.

  Returns:
    A new Metadata message with entries removed and the same
      fingerprint as existing_metadata if existing_metadata contains
      a fingerprint.
  """
    if remove_all:
        new_metadata_message = message_classes.Metadata()
    elif keys:
        existing_metadata_dict = _MetadataMessageToDict(existing_metadata)
        for key in keys:
            existing_metadata_dict.pop(key, None)
        new_metadata_message = _DictToMetadataMessage(message_classes, existing_metadata_dict)
    new_metadata_message.fingerprint = existing_metadata.fingerprint
    return new_metadata_message