from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GKEMasterInfo(_messages.Message):
    """For display only. Metadata associated with a Google Kubernetes Engine
  (GKE) cluster master.

  Fields:
    clusterNetworkUri: URI of a GKE cluster network.
    clusterUri: URI of a GKE cluster.
    externalIp: External IP address of a GKE cluster master.
    internalIp: Internal IP address of a GKE cluster master.
  """
    clusterNetworkUri = _messages.StringField(1)
    clusterUri = _messages.StringField(2)
    externalIp = _messages.StringField(3)
    internalIp = _messages.StringField(4)