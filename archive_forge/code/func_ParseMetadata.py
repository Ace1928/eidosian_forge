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
def ParseMetadata(metadata, metadata_from_file, messages=None):
    """Parse and create metadata object from the parsed arguments.

  Args:
    metadata: dict, key-value pairs passed in from the --metadata flag.
    metadata_from_file: dict, key-path pairs passed in from  the
      --metadata-from-file flag.
    messages: module or None, the apitools messages module for Cloud IoT (uses a
      default module if not provided).

  Returns:
    MetadataValue or None, the populated metadata message for a Device.

  Raises:
    InvalidMetadataError: if there was any issue parsing the metadata.
  """
    if not metadata and (not metadata_from_file):
        return None
    metadata = metadata or dict()
    metadata_from_file = metadata_from_file or dict()
    if len(metadata) + len(metadata_from_file) > MAX_METADATA_PAIRS:
        raise InvalidMetadataError('Maximum number of metadata key-value pairs is {}.'.format(MAX_METADATA_PAIRS))
    if set(metadata.keys()) & set(metadata_from_file.keys()):
        raise InvalidMetadataError('Cannot specify the same key in both --metadata and --metadata-from-file.')
    total_size = 0
    messages = messages or devices.GetMessagesModule()
    additional_properties = []
    for key, value in six.iteritems(metadata):
        total_size += len(key) + len(value)
        additional_properties.append(_ValidateAndCreateAdditionalProperty(messages, key, value))
    for key, path in metadata_from_file.items():
        value = _ReadMetadataValueFromFile(path)
        total_size += len(key) + len(value)
        additional_properties.append(_ValidateAndCreateAdditionalProperty(messages, key, value))
    if total_size > MAX_METADATA_SIZE:
        raise InvalidMetadataError('Maximum size of metadata key-value pairs is 256KB.')
    return messages.Device.MetadataValue(additionalProperties=additional_properties)