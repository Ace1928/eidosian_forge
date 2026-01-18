from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SetAgentUri(unused_instance_ref, args, request):
    dest = args.destination
    if IsBucketUri(dest) and storage_util.ValidateBucketUrl:
        request.googleCloudDialogflowV2ExportAgentRequest = {'agentUri': dest}
    return request