from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsIapTunnelLocationsDestGroupsPatchRequest(_messages.Message):
    """A IapProjectsIapTunnelLocationsDestGroupsPatchRequest object.

  Fields:
    name: Required. Immutable. Identifier for the TunnelDestGroup. Must be
      unique within the project and contain only lower case letters (a-z) and
      dashes (-).
    tunnelDestGroup: A TunnelDestGroup resource to be passed as the request
      body.
    updateMask: A field mask that specifies which IAP settings to update. If
      omitted, then all of the settings are updated. See
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    name = _messages.StringField(1, required=True)
    tunnelDestGroup = _messages.MessageField('TunnelDestGroup', 2)
    updateMask = _messages.StringField(3)