from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BestEffortProvisioning(_messages.Message):
    """Best effort provisioning.

  Fields:
    enabled: When this is enabled, cluster/node pool creations will ignore
      non-fatal errors like stockout to best provision as many nodes as
      possible right now and eventually bring up all target number of nodes
    minProvisionNodes: Minimum number of nodes to be provisioned to be
      considered as succeeded, and the rest of nodes will be provisioned
      gradually and eventually when stockout issue has been resolved.
  """
    enabled = _messages.BooleanField(1)
    minProvisionNodes = _messages.IntegerField(2, variant=_messages.Variant.INT32)