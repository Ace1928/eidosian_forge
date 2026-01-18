from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadMetadataConfig(_messages.Message):
    """WorkloadMetadataConfig defines the metadata configuration to expose to
  workloads on the node pool.

  Enums:
    ModeValueValuesEnum: Mode is the configuration for how to expose metadata
      to workloads running on the node pool.

  Fields:
    mode: Mode is the configuration for how to expose metadata to workloads
      running on the node pool.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Mode is the configuration for how to expose metadata to workloads
    running on the node pool.

    Values:
      MODE_UNSPECIFIED: Not set.
      GCE_METADATA: Expose all Compute Engine metadata to pods.
      GKE_METADATA: Run the GKE Metadata Server on this node. The GKE Metadata
        Server exposes a metadata API to workloads that is compatible with the
        V1 Compute Metadata APIs exposed by the Compute Engine and App Engine
        Metadata Servers. This feature can only be enabled if Workload
        Identity is enabled at the cluster level.
    """
        MODE_UNSPECIFIED = 0
        GCE_METADATA = 1
        GKE_METADATA = 2
    mode = _messages.EnumField('ModeValueValuesEnum', 1)