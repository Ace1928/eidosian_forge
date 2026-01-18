from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def UpdateRequestLangDetection(unused_instance_ref, args, request):
    """The hook to inject content into the language detection request."""
    content = args.content
    content_file = args.content_file
    messages = apis.GetMessagesModule(SPEECH_API, _GetApiVersion(args))
    detect_language_request = messages.DetectLanguageRequest()
    project = properties.VALUES.core.project.GetOrFail()
    request.parent = 'projects/{}/locations/{}'.format(project, args.zone)
    if args.IsSpecified('model'):
        project = properties.VALUES.core.project.GetOrFail()
        model = 'projects/{}/locations/{}/models/language-detection/{}'.format(project, args.zone, args.model)
        detect_language_request.model = model
    if content_file:
        if os.path.isfile(content_file):
            detect_language_request.content = files.ReadFileContents(content_file)
        else:
            raise ContentFileError('Could not find --content-file [{}]. Content file must be a path to a local file)'.format(content_file))
    else:
        detect_language_request.content = content
    if args.IsSpecified('mime_type'):
        detect_language_request.mimeType = args.mime_type
    request.detectLanguageRequest = detect_language_request
    return request