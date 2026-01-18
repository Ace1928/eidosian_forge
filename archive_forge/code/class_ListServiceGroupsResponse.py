from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceGroupsResponse(_messages.Message):
    """The response message of the `ListServiceGroups` method.

  Fields:
    groups: The group states exposed by the parent service.
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    groups = _messages.MessageField('GroupState', 1, repeated=True)
    nextPageToken = _messages.StringField(2)