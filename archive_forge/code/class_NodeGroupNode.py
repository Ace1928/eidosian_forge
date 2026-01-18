from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupNode(_messages.Message):
    """A NodeGroupNode object.

  Enums:
    CpuOvercommitTypeValueValuesEnum: CPU overcommit.
    StatusValueValuesEnum:

  Fields:
    accelerators: Accelerators for this node.
    consumedResources: Node resources that are reserved by all instances.
    cpuOvercommitType: CPU overcommit.
    disks: Local disk configurations.
    instanceConsumptionData: Instance data that shows consumed resources on
      the node.
    instances: Instances scheduled on this node.
    name: The name of the node.
    nodeType: The type of this node.
    satisfiesPzs: [Output Only] Reserved for future use.
    serverBinding: Binding properties for the physical server.
    serverId: Server ID associated with this node.
    status: A StatusValueValuesEnum attribute.
    totalResources: Total amount of available resources on the node.
    upcomingMaintenance: [Output Only] The information about an upcoming
      maintenance event.
  """

    class CpuOvercommitTypeValueValuesEnum(_messages.Enum):
        """CPU overcommit.

    Values:
      CPU_OVERCOMMIT_TYPE_UNSPECIFIED: <no description>
      ENABLED: <no description>
      NONE: <no description>
    """
        CPU_OVERCOMMIT_TYPE_UNSPECIFIED = 0
        ENABLED = 1
        NONE = 2

    class StatusValueValuesEnum(_messages.Enum):
        """StatusValueValuesEnum enum type.

    Values:
      CREATING: <no description>
      DELETING: <no description>
      INVALID: <no description>
      READY: <no description>
      REPAIRING: <no description>
    """
        CREATING = 0
        DELETING = 1
        INVALID = 2
        READY = 3
        REPAIRING = 4
    accelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    consumedResources = _messages.MessageField('InstanceConsumptionInfo', 2)
    cpuOvercommitType = _messages.EnumField('CpuOvercommitTypeValueValuesEnum', 3)
    disks = _messages.MessageField('LocalDisk', 4, repeated=True)
    instanceConsumptionData = _messages.MessageField('InstanceConsumptionData', 5, repeated=True)
    instances = _messages.StringField(6, repeated=True)
    name = _messages.StringField(7)
    nodeType = _messages.StringField(8)
    satisfiesPzs = _messages.BooleanField(9)
    serverBinding = _messages.MessageField('ServerBinding', 10)
    serverId = _messages.StringField(11)
    status = _messages.EnumField('StatusValueValuesEnum', 12)
    totalResources = _messages.MessageField('InstanceConsumptionInfo', 13)
    upcomingMaintenance = _messages.MessageField('UpcomingMaintenance', 14)