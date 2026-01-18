from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ResourcePool(_messages.Message):
    """Represents the spec of a group of resources of the same type, for
  example machine type, disk, and accelerators, in a PersistentResource.

  Fields:
    autoscalingSpec: Optional. Optional spec to configure GKE autoscaling
    diskSpec: Optional. Disk spec for the machine in this node pool.
    id: Immutable. The unique ID in a PersistentResource for referring to this
      resource pool. User can specify it if necessary. Otherwise, it's
      generated automatically.
    machineSpec: Required. Immutable. The specification of a single machine.
    replicaCount: Optional. The total number of machines to use for this
      resource pool.
    usedReplicaCount: Output only. The number of machines currently in use by
      training jobs for this resource pool. Will replace idle_replica_count.
  """
    autoscalingSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1ResourcePoolAutoscalingSpec', 1)
    diskSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1DiskSpec', 2)
    id = _messages.StringField(3)
    machineSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1MachineSpec', 4)
    replicaCount = _messages.IntegerField(5)
    usedReplicaCount = _messages.IntegerField(6)