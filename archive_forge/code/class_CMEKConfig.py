from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CMEKConfig(_messages.Message):
    """CMEKConfig holds the resource address to KMS keys or grants which are
  used for signing certs and token that are used for communication within
  cluster.

  Fields:
    cmekAggregationCaKey: Resource address to aggregation CA's root key
      managed in KMS.
    cmekClusterCaKey: Resource address to cluster CA's root key managed in
      KMS.
    cmekControlPlaneDisksCaKey: Resource address to control plane CA's root
      key managed in KMS.
    cmekEtcdApiServerCaKey: Resource address to etcd<->apiserver CA's root key
      managed in KMS.
    cmekEtcdPeerCaKey: Resource address to etcd peer CA's root key managed in
      KMS.
    cmekServiceAccountKey: Resource address to service account key managed in
      KMS.
  """
    cmekAggregationCaKey = _messages.StringField(1)
    cmekClusterCaKey = _messages.StringField(2)
    cmekControlPlaneDisksCaKey = _messages.StringField(3)
    cmekEtcdApiServerCaKey = _messages.StringField(4)
    cmekEtcdPeerCaKey = _messages.StringField(5)
    cmekServiceAccountKey = _messages.StringField(6)