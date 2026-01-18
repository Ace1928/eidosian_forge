from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupsListInstancesRequest(_messages.Message):
    """A RegionInstanceGroupsListInstancesRequest object.

  Enums:
    InstanceStateValueValuesEnum: Instances in which state should be returned.
      Valid options are: 'ALL', 'RUNNING'. By default, it lists all instances.

  Fields:
    instanceState: Instances in which state should be returned. Valid options
      are: 'ALL', 'RUNNING'. By default, it lists all instances.
    portName: Name of port user is interested in. It is optional. If it is
      set, only information about this ports will be returned. If it is not
      set, all the named ports will be returned. Always lists all instances.
  """

    class InstanceStateValueValuesEnum(_messages.Enum):
        """Instances in which state should be returned. Valid options are: 'ALL',
    'RUNNING'. By default, it lists all instances.

    Values:
      ALL: Matches any status of the instances, running, non-running and
        others.
      RUNNING: Instance is in RUNNING state if it is running.
    """
        ALL = 0
        RUNNING = 1
    instanceState = _messages.EnumField('InstanceStateValueValuesEnum', 1)
    portName = _messages.StringField(2)