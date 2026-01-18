from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KubernetesFeature(_messages.Message):
    """For Edge Appliance, a feature that allows the appliance to run a
  Kubernetes node.

  Fields:
    serviceAccount: Optional. Name of the cluster registration service
      account. Follow the steps mentioned here
      (https://cloud.google.com/distributed-cloud/edge-
      appliance/docs/configure-cloud#cluster_sa) before setting this field.
  """
    serviceAccount = _messages.StringField(1)