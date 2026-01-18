from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChromeOsDevices(_messages.Message):
    """JSON response template for List Chrome OS Devices operation in Directory

  API.

  Fields:
    chromeosdevices: List of Chrome OS Device objects.
    etag: ETag of the resource.
    kind: Kind of resource this is.
    nextPageToken: Token used to access next page of this result.
  """
    chromeosdevices = _messages.MessageField('ChromeOsDevice', 1, repeated=True)
    etag = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'admin#directory#chromeosdevices')
    nextPageToken = _messages.StringField(4)