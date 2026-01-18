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
def _DictToMetadataMessage(message_classes, metadata_dict):
    """Converts a metadata dict to a Metadata message."""
    message = message_classes.Metadata()
    if metadata_dict:
        for key, value in sorted(six.iteritems(metadata_dict)):
            message.items.append(message_classes.Metadata.ItemsValueListEntry(key=key, value=value))
    return message