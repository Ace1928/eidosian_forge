from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1WorkloadIdentityConfig(_messages.Message):
    """Workload Identity settings.

  Fields:
    identityProvider: The ID of the OIDC Identity Provider (IdP) associated to
      the Workload Identity Pool.
    issuerUri: The OIDC issuer URL for this cluster.
    workloadPool: The Workload Identity Pool associated to the cluster.
  """
    identityProvider = _messages.StringField(1)
    issuerUri = _messages.StringField(2)
    workloadPool = _messages.StringField(3)