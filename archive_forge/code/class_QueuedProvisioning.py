from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueuedProvisioning(_messages.Message):
    """QueuedProvisioning defines the queued provisioning used by the node
  pool.

  Fields:
    enabled: Denotes that this nodepool is QRM specific, meaning nodes can be
      only obtained through queuing via the Cluster Autoscaler
      ProvisioningRequest API.
  """
    enabled = _messages.BooleanField(1)