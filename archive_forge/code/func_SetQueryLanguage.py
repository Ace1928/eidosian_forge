from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SetQueryLanguage(unused_instance_ref, args, request):
    query_input = request.googleCloudDialogflowV2DetectIntentRequest.queryInput
    if args.IsSpecified('query_text'):
        query_input.text.languageCode = args.language
    elif args.IsSpecified('query_audio_file'):
        query_input.audioConfig.languageCode = args.language
    return request