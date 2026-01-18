from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPrivateCloudsResponse(_messages.Message):
    """Response message for VmwareEngine.ListPrivateClouds

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    privateClouds: A list of private clouds.
    unreachable: Locations that could not be reached when making an aggregated
      query using wildcards.
  """
    nextPageToken = _messages.StringField(1)
    privateClouds = _messages.MessageField('PrivateCloud', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)