from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListOrganizationsResponse(_messages.Message):
    """The response returned from the `ListOrganizations` method.

  Fields:
    nextPageToken: A pagination token to be used to retrieve the next page of
      results. If the result is too large to fit within the page size
      specified in the request, this field will be set with a token that can
      be used to fetch the next page of results. If this field is empty, it
      indicates that this response contains the last page of results.
    organizations: The list of Organizations that matched the list query,
      possibly paginated.
  """
    nextPageToken = _messages.StringField(1)
    organizations = _messages.MessageField('Organization', 2, repeated=True)