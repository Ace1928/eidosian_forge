from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def UpdateRequestGetSupportedLanguages(unused_instance_ref, args, request):
    """The hook to inject content into the getSupportedLanguages request."""
    project = properties.VALUES.core.project.GetOrFail()
    request.parent = 'projects/{}/locations/{}'.format(project, args.zone)
    if args.IsSpecified('model'):
        model = 'projects/{}/locations/{}/models/{}'.format(project, args.zone, args.model)
        request.model = model
    return request