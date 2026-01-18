from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalWorkloadNodeConfig(_messages.Message):
    """Specifies the workload node configurations.

  Enums:
    ContainerRuntimeValueValuesEnum: Specifies which container runtime will be
      used.

  Fields:
    containerRuntime: Specifies which container runtime will be used.
    maxPodsPerNode: The maximum number of pods a node can run. The size of the
      CIDR range assigned to the node will be derived from this parameter.
  """

    class ContainerRuntimeValueValuesEnum(_messages.Enum):
        """Specifies which container runtime will be used.

    Values:
      CONTAINER_RUNTIME_UNSPECIFIED: No container runtime selected.
      CONTAINERD: Containerd runtime.
    """
        CONTAINER_RUNTIME_UNSPECIFIED = 0
        CONTAINERD = 1
    containerRuntime = _messages.EnumField('ContainerRuntimeValueValuesEnum', 1)
    maxPodsPerNode = _messages.IntegerField(2)