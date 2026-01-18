from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnhancedKCPIngress(_messages.Message):
    """Enhanced KCP ingress configuration.

  Fields:
    enabled: Controls whether the cluster is configured to use enhanced KCP
      ingress.
    endpoint: Output only. The cluster's DNS endpoint configuration. A DNS
      format address. This is accessible from the public internet. Ex: uid.us-
      central1.gke.goog.
  """
    enabled = _messages.BooleanField(1)
    endpoint = _messages.StringField(2)