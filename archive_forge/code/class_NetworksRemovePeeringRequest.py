from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksRemovePeeringRequest(_messages.Message):
    """A NetworksRemovePeeringRequest object.

  Fields:
    name: Name of the peering, which should conform to RFC1035.
  """
    name = _messages.StringField(1)