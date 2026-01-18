from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def ModifyImportOrRestoreRequest(unused_instance_ref, args, request):
    body = _GetAgentRequestBody(args.source)
    if args.replace_all:
        request.googleCloudDialogflowV2RestoreAgentRequest = body
    else:
        request.googleCloudDialogflowV2ImportAgentRequest = body
    return request