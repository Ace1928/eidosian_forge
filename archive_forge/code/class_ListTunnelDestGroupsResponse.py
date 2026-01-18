from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTunnelDestGroupsResponse(_messages.Message):
    """The response from ListTunnelDestGroups.

  Fields:
    nextPageToken: A token that you can send as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    tunnelDestGroups: TunnelDestGroup existing in the project.
  """
    nextPageToken = _messages.StringField(1)
    tunnelDestGroups = _messages.MessageField('TunnelDestGroup', 2, repeated=True)