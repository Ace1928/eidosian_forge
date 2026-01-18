from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryTestablePermissionsResponse(_messages.Message):
    """The response containing permissions which can be tested on a resource.

  Fields:
    nextPageToken: To retrieve the next page of results, set
      `QueryTestableRolesRequest.page_token` to this value.
    permissions: The Permissions testable on the requested resource.
  """
    nextPageToken = _messages.StringField(1)
    permissions = _messages.MessageField('Permission', 2, repeated=True)