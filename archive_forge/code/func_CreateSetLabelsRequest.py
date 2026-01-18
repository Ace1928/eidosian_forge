from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateSetLabelsRequest(args, messages):
    instance = GetInstanceResource(args).RelativeName()
    set_label_request = messages.SetInstanceLabelsRequest()
    labels_message = messages.SetInstanceLabelsRequest.LabelsValue
    set_label_request.labels = labels_message(additionalProperties=[labels_message.AdditionalProperty(key=key, value=value) for key, value in args.labels.items()])
    return messages.NotebooksProjectsLocationsInstancesSetLabelsRequest(name=instance, setInstanceLabelsRequest=set_label_request)