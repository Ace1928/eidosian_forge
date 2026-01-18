from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListEnvironmentsResponse(_messages.Message):
    """The response message for Environments.ListEnvironments.

  Fields:
    environments: The list of agent environments. There will be a maximum
      number of items returned based on the page_size field in the request.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    environments = _messages.MessageField('GoogleCloudDialogflowV2Environment', 1, repeated=True)
    nextPageToken = _messages.StringField(2)