from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityActionsResponse(_messages.Message):
    """Contains a list of SecurityActions in response to a
  ListSecurityActionRequest.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    securityActions: The SecurityActions for the specified environment.
  """
    nextPageToken = _messages.StringField(1)
    securityActions = _messages.MessageField('GoogleCloudApigeeV1SecurityAction', 2, repeated=True)