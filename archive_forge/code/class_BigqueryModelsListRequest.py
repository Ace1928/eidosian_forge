from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryModelsListRequest(_messages.Message):
    """A BigqueryModelsListRequest object.

  Fields:
    datasetId: Required. Dataset ID of the models to list.
    maxResults: The maximum number of results to return in a single response
      page. Leverage the page tokens to iterate through the entire collection.
    pageToken: Page token, returned by a previous call to request the next
      page of results
    projectId: Required. Project ID of the models to list.
  """
    datasetId = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)