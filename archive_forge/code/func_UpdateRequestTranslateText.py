from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def UpdateRequestTranslateText(unused_instance_ref, args, request):
    """The hook to inject content into the translate request."""
    content = args.content
    content_file = args.content_file
    messages = apis.GetMessagesModule(SPEECH_API, _GetApiVersion(args))
    translate_text_request = messages.TranslateTextRequest()
    project = properties.VALUES.core.project.GetOrFail()
    request.parent = 'projects/{}/locations/{}'.format(project, args.zone)
    if args.IsSpecified('model'):
        project = properties.VALUES.core.project.GetOrFail()
        model = 'projects/{}/locations/{}/models/{}'.format(project, args.zone, args.model)
        translate_text_request.model = model
    if content_file:
        if os.path.isfile(content_file):
            translate_text_request.contents = [files.ReadFileContents(content_file)]
        else:
            raise ContentFileError('Could not find --content-file [{}]. Content file must be a path to a local file)'.format(content_file))
    else:
        translate_text_request.contents = [content]
    if args.IsSpecified('mime_type'):
        translate_text_request.mimeType = args.mime_type
    if args.IsSpecified('glossary_config'):
        translate_text_request.glossaryConfig = messages.TranslateTextGlossaryConfig(glossary=args.glossaryConfig)
    if args.IsSpecified('source_language'):
        translate_text_request.sourceLanguageCode = args.source_language
    translate_text_request.targetLanguageCode = args.target_language
    request.translateTextRequest = translate_text_request
    return request