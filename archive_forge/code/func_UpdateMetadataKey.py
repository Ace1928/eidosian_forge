from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def UpdateMetadataKey(self, metadata, key, value):
    """Updates a key in the TPU metadata object.

    If the key does not exist, it is added.

    Args:
      metadata: tpu.messages.Node.MetadataValue, the TPU's metadata.
      key: str, the key to be updated.
      value: str, the new value for the key.

    Returns:
      The updated metadata.
    """
    if metadata is None or metadata.additionalProperties is None:
        return self.messages.Node.MetadataValue(additionalProperties=[self.messages.Node.MetadataValue.AdditionalProperty(key=key, value=value)])
    item = None
    for x in metadata.additionalProperties:
        if x.key == key:
            item = x
            break
    if item is not None:
        item.value = value
    else:
        metadata.additionalProperties.append(self.messages.Node.MetadataValue.AdditionalProperty(key=key, value=value))
    return metadata