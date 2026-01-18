from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustedApps(_messages.Message):
    """JSON template for Trusted Apps response object of a user in Directory

  API.

  Fields:
    etag: ETag of the resource.
    kind: Identifies the resource as trusted apps response.
    nextPageToken: A string attribute.
    trustedApps: Trusted Apps list.
  """
    etag = _messages.StringField(1)
    kind = _messages.StringField(2, default=u'admin#directory#trustedapplist')
    nextPageToken = _messages.StringField(3)
    trustedApps = _messages.MessageField('TrustedAppId', 4, repeated=True)