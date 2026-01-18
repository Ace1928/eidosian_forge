from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Subnetwork(_messages.Message):
    """Message returning the created service subnetwork.

  Fields:
    ipCidrRange: Subnetwork CIDR range in "10.x.x.x/y" format.
    name: Subnetwork name. See https://cloud.google.com/compute/docs/vpc/
  """
    ipCidrRange = _messages.StringField(1)
    name = _messages.StringField(2)