from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetLabelsFromArgs(args, messages):
    if args.IsSpecified('labels'):
        labels_message = messages.Instance.LabelsValue
        return labels_message(additionalProperties=[labels_message.AdditionalProperty(key=key, value=value) for key, value in args.labels.items()])
    return None