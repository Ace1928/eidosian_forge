from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PacketMirroringMirroredResourceInfoInstanceInfo(_messages.Message):
    """A PacketMirroringMirroredResourceInfoInstanceInfo object.

  Fields:
    canonicalUrl: [Output Only] Unique identifier for the instance; defined by
      the server.
    url: Resource URL to the virtual machine instance which is being mirrored.
  """
    canonicalUrl = _messages.StringField(1)
    url = _messages.StringField(2)