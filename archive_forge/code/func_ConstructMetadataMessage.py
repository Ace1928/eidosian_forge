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
def ConstructMetadataMessage(message_classes, metadata=None, metadata_from_file=None, existing_metadata=None):
    """Creates a Metadata message from the given dicts of metadata.

  Args:
    message_classes: An object containing API message classes.
    metadata: A dict mapping metadata keys to metadata values or None.
    metadata_from_file: A dict mapping metadata keys to file names containing
      the keys' values or None.
    existing_metadata: If not None, the given metadata values are combined with
      this Metadata message.

  Raises:
    ToolException: If metadata and metadata_from_file contain duplicate
      keys or if there is a problem reading the contents of a file in
      metadata_from_file.

  Returns:
    A Metadata protobuf.
  """
    new_metadata_dict = ConstructMetadataDict(metadata, metadata_from_file)
    existing_metadata_dict = _MetadataMessageToDict(existing_metadata)
    existing_metadata_dict.update(new_metadata_dict)
    try:
        _ValidateSshKeys(existing_metadata_dict)
    except InvalidSshKeyException as e:
        log.warning(e)
    new_metadata_message = _DictToMetadataMessage(message_classes, existing_metadata_dict)
    if existing_metadata:
        new_metadata_message.fingerprint = existing_metadata.fingerprint
    return new_metadata_message