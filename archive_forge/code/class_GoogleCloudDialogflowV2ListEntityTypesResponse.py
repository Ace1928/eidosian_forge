from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListEntityTypesResponse(_messages.Message):
    """The response message for EntityTypes.ListEntityTypes.

  Fields:
    entityTypes: The list of agent entity types. There will be a maximum
      number of items returned based on the page_size field in the request.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    entityTypes = _messages.MessageField('GoogleCloudDialogflowV2EntityType', 1, repeated=True)
    nextPageToken = _messages.StringField(2)