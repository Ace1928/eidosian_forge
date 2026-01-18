from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryItem(_messages.Message):
    """A single piece of inventory on a VM.

  Enums:
    OriginTypeValueValuesEnum: The origin of this inventory item.
    TypeValueValuesEnum: The specific type of inventory, correlating to its
      specific details.

  Fields:
    availablePackage: Software package available to be installed on the VM
      instance.
    createTime: When this inventory item was first detected.
    id: Identifier for this item, unique across items for this VM.
    installedPackage: Software package present on the VM instance.
    originType: The origin of this inventory item.
    type: The specific type of inventory, correlating to its specific details.
    updateTime: When this inventory item was last modified.
  """

    class OriginTypeValueValuesEnum(_messages.Enum):
        """The origin of this inventory item.

    Values:
      ORIGIN_TYPE_UNSPECIFIED: Invalid. An origin type must be specified.
      INVENTORY_REPORT: This inventory item was discovered as the result of
        the agent reporting inventory via the reporting API.
    """
        ORIGIN_TYPE_UNSPECIFIED = 0
        INVENTORY_REPORT = 1

    class TypeValueValuesEnum(_messages.Enum):
        """The specific type of inventory, correlating to its specific details.

    Values:
      TYPE_UNSPECIFIED: Invalid. An type must be specified.
      INSTALLED_PACKAGE: This represents a package that is installed on the
        VM.
      AVAILABLE_PACKAGE: This represents an update that is available for a
        package.
    """
        TYPE_UNSPECIFIED = 0
        INSTALLED_PACKAGE = 1
        AVAILABLE_PACKAGE = 2
    availablePackage = _messages.MessageField('InventorySoftwarePackage', 1)
    createTime = _messages.StringField(2)
    id = _messages.StringField(3)
    installedPackage = _messages.MessageField('InventorySoftwarePackage', 4)
    originType = _messages.EnumField('OriginTypeValueValuesEnum', 5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)
    updateTime = _messages.StringField(7)