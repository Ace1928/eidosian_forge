from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PscConnection(_messages.Message):
    """Details of consumer resources in a PSC connection.

  Fields:
    address: Output only. The IP allocated on the consumer network for the PSC
      forwarding rule.
    forwardingRule: Output only. The URI of the consumer side forwarding rule.
      Example: projects/{projectNumOrId}/regions/us-
      east1/forwardingRules/{resourceId}.
    network: The consumer network where the IP address resides, in the form of
      projects/{project_id}/global/networks/{network_id}.
    projectId: Output only. The consumer project_id where the forwarding rule
      is created from.
    pscConnectionId: Output only. The PSC connection id of the forwarding rule
      connected to the service attachment.
  """
    address = _messages.StringField(1)
    forwardingRule = _messages.StringField(2)
    network = _messages.StringField(3)
    projectId = _messages.StringField(4)
    pscConnectionId = _messages.StringField(5)