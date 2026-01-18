from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalIslandModeCidrConfig(_messages.Message):
    """Specifies the cluster CIDR configuration while running in island mode.

  Fields:
    podAddressCidrBlocks: Required. All pods in the cluster are assigned an
      RFC1918 IPv4 address from these ranges. This field cannot be changed
      after creation.
    serviceAddressCidrBlocks: Required. All services in the cluster are
      assigned an RFC1918 IPv4 address from these ranges. This field is
      mutable after creation starting with version 1.15.
  """
    podAddressCidrBlocks = _messages.StringField(1, repeated=True)
    serviceAddressCidrBlocks = _messages.StringField(2, repeated=True)