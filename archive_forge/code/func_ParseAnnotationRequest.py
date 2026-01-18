from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
def ParseAnnotationRequest(ref, args, req):
    """Prepare the annotation for create and update requests."""
    del ref
    messages = apis.GetMessagesModule('mediaasset', 'v1alpha')
    if req.annotation is None:
        req.annotation = encoding.DictToMessage({}, messages.Annotation)
    if args.IsKnownAndSpecified('labels'):
        req.annotation.labels = encoding.DictToMessage(args.labels, messages.Annotation.LabelsValue)
    if args.IsKnownAndSpecified('annotation_data_file'):
        annotation_data = json.loads(args.annotation_data_file)
        req.annotation.data = encoding.DictToMessage(annotation_data, messages.Annotation.DataValue)
    return req