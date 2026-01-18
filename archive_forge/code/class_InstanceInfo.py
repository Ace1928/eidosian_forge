from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceInfo(_messages.Message):
    """For display only. Metadata associated with a Compute Engine instance.

  Fields:
    displayName: Name of a Compute Engine instance.
    externalIp: External IP address of the network interface.
    interface: Name of the network interface of a Compute Engine instance.
    internalIp: Internal IP address of the network interface.
    networkTags: Network tags configured on the instance.
    networkUri: URI of a Compute Engine network.
    serviceAccount: Service account authorized for the instance.
    uri: URI of a Compute Engine instance.
  """
    displayName = _messages.StringField(1)
    externalIp = _messages.StringField(2)
    interface = _messages.StringField(3)
    internalIp = _messages.StringField(4)
    networkTags = _messages.StringField(5, repeated=True)
    networkUri = _messages.StringField(6)
    serviceAccount = _messages.StringField(7)
    uri = _messages.StringField(8)