from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateIPv6Status(_messages.Message):
    """PrivateIPv6Status contains the desired state of the IPv6 fast path on
  this cluster. Private IPv6 access allows direct high speed communication
  from GKE pods to gRPC Google cloud services over IPv6.

  Fields:
    enabled: Enables private IPv6 access to Google Cloud services for this
      cluster.
  """
    enabled = _messages.BooleanField(1)