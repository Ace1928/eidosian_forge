from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListModelsResponse(_messages.Message):
    """Response message for the ListModels method.

  Fields:
    models: The list of models.
    nextPageToken: Optional. Pass this token as the `page_token` field of the
      request for a subsequent call.
  """
    models = _messages.MessageField('GoogleCloudMlV1Model', 1, repeated=True)
    nextPageToken = _messages.StringField(2)