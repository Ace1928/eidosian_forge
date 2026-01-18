from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceNetworking(_messages.Message):
    """Information about the Kubernetes Service networking configuration.

  Fields:
    deployment: Required. Name of the Kubernetes Deployment whose traffic is
      managed by the specified Service.
    disablePodOverprovisioning: Optional. Whether to disable Pod
      overprovisioning. If Pod overprovisioning is disabled then Cloud Deploy
      will limit the number of total Pods used for the deployment strategy to
      the number of Pods the Deployment has on the cluster.
    service: Required. Name of the Kubernetes Service.
  """
    deployment = _messages.StringField(1)
    disablePodOverprovisioning = _messages.BooleanField(2)
    service = _messages.StringField(3)