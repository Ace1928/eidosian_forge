from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAnnotationsResponse(_messages.Message):
    """Response message for AnnotationsService.ListAnnotations.

  Fields:
    annotations: The annotations from the specified asset.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    annotations = _messages.MessageField('Annotation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)