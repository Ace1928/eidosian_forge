from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoveryEndpoint(_messages.Message):
    """Endpoints on each network, for Redis clients to connect to the cluster.

  Fields:
    address: Output only. Address of the exposed Redis endpoint used by
      clients to connect to the service. The address could be either IP or
      hostname.
    port: Output only. The port number of the exposed Redis endpoint.
    pscConfig: Output only. Customer configuration for where the endpoint is
      created and accessed from.
  """
    address = _messages.StringField(1)
    port = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pscConfig = _messages.MessageField('PscConfig', 3)