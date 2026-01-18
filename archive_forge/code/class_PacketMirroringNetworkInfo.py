from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringNetworkInfo(_messages.Message):
    """A PacketMirroringNetworkInfo object.

  Fields:
    canonicalUrl: [Output Only] Unique identifier for the network; defined by
      the server.
    url: URL of the network resource.
  """
    canonicalUrl = _messages.StringField(1)
    url = _messages.StringField(2)