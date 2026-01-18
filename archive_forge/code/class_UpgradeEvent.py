from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeEvent(_messages.Message):
    """UpgradeEvent is a notification sent to customers by the cluster server
  when a resource is upgrading.

  Enums:
    ResourceTypeValueValuesEnum: The resource type that is upgrading.

  Fields:
    currentVersion: The current version before the upgrade.
    operation: The operation associated with this upgrade.
    operationStartTime: The time when the operation was started.
    resource: Optional relative path to the resource. For example in node pool
      upgrades, the relative path of the node pool.
    resourceType: The resource type that is upgrading.
    targetVersion: The target version for the upgrade.
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """The resource type that is upgrading.

    Values:
      UPGRADE_RESOURCE_TYPE_UNSPECIFIED: Default value. This shouldn't be
        used.
      MASTER: Master / control plane
      NODE_POOL: Node pool
    """
        UPGRADE_RESOURCE_TYPE_UNSPECIFIED = 0
        MASTER = 1
        NODE_POOL = 2
    currentVersion = _messages.StringField(1)
    operation = _messages.StringField(2)
    operationStartTime = _messages.StringField(3)
    resource = _messages.StringField(4)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 5)
    targetVersion = _messages.StringField(6)