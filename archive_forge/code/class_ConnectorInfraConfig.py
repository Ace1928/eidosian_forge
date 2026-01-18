from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorInfraConfig(_messages.Message):
    """This cofiguration provides infra configs like rate limit threshold which
  need to be configurable for every connector version

  Fields:
    internalclientRatelimitThreshold: Max QPS supported for internal requests
      originating from Connd.
    ratelimitThreshold: Max QPS supported by the connector version before
      throttling of requests.
  """
    internalclientRatelimitThreshold = _messages.IntegerField(1)
    ratelimitThreshold = _messages.IntegerField(2)