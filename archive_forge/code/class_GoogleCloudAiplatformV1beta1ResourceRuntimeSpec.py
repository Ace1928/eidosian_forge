from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ResourceRuntimeSpec(_messages.Message):
    """Configuration for the runtime on a PersistentResource instance,
  including but not limited to: * Service accounts used to run the workloads.
  * Whether to make it a dedicated Ray Cluster.

  Fields:
    raySpec: Optional. Ray cluster configuration. Required when creating a
      dedicated RayCluster on the PersistentResource.
    serviceAccountSpec: Optional. Configure the use of workload identity on
      the PersistentResource
  """
    raySpec = _messages.MessageField('GoogleCloudAiplatformV1beta1RaySpec', 1)
    serviceAccountSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ServiceAccountSpec', 2)