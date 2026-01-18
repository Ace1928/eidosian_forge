from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListAnnotationsResponse(_messages.Message):
    """Response message for DatasetService.ListAnnotations.

  Fields:
    annotations: A list of Annotations that matches the specified filter in
      the request.
    nextPageToken: The standard List next-page token.
  """
    annotations = _messages.MessageField('GoogleCloudAiplatformV1beta1Annotation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)