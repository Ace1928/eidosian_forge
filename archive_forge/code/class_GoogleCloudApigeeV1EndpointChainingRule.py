from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EndpointChainingRule(_messages.Message):
    """EndpointChainingRule specifies the proxies contained in a particular
  deployment group, so that other deployment groups can find them in chaining
  calls.

  Fields:
    deploymentGroup: The deployment group to target for cross-shard chaining
      calls to these proxies.
    proxyIds: List of proxy ids which may be found in the given deployment
      group.
  """
    deploymentGroup = _messages.StringField(1)
    proxyIds = _messages.StringField(2, repeated=True)