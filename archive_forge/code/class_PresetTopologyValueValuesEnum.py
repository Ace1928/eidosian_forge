from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PresetTopologyValueValuesEnum(_messages.Enum):
    """Optional. The topology implemented in this hub. Currently, this field
    is only used when policy_mode = PRESET. The available preset topologies
    are MESH and STAR. If preset_topology is unspecified and policy_mode =
    PRESET, the preset_topology defaults to MESH. When policy_mode = CUSTOM,
    the preset_topology is set to PRESET_TOPOLOGY_UNSPECIFIED.

    Values:
      PRESET_TOPOLOGY_UNSPECIFIED: Preset topology is unspecified. When
        policy_mode = PRESET, it defaults to MESH.
      PRESET_TOPOLOGY_DISALLOWED: No preset topology is allowed. It is used
        when policy_mode is `custom`.
      MESH: Mesh topology is implemented. Group `default` is automatically
        created. All spokes in the hub are added to group `default`.
      STAR: Star topology is implemented. Two groups, `center` and `edge`, are
        automatically created along with hub creation. Spokes have to join one
        of the groups during creation.
    """
    PRESET_TOPOLOGY_UNSPECIFIED = 0
    PRESET_TOPOLOGY_DISALLOWED = 1
    MESH = 2
    STAR = 3