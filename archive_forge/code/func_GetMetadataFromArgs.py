from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetMetadataFromArgs(args, messages):
    if args.IsSpecified('metadata'):
        metadata_message = messages.GceSetup.MetadataValue
        return metadata_message(additionalProperties=[metadata_message.AdditionalProperty(key=key, value=value) for key, value in args.metadata.items()])
    return None