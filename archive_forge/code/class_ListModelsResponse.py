from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListModelsResponse(_messages.Message):
    """Response message for ListModels.

  Fields:
    models: The models read.
    nextPageToken: A token to retrieve next page of results. Pass this token
      to the page_token field in the ListModelsRequest to obtain the
      corresponding page.
  """
    models = _messages.MessageField('Model', 1, repeated=True)
    nextPageToken = _messages.StringField(2)