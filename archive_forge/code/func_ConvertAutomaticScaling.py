from __future__ import absolute_import
import re
def ConvertAutomaticScaling(automatic_scaling):
    """Moves several VM-specific automatic scaling parameters to submessages.

  For example:
  Input {
    "targetSentPacketsPerSec": 10,
    "targetReadOpsPerSec": 2,
    "targetRequestCountPerSec": 3
  }
  Output {
    "networkUtilization": {
      "targetSentPacketsPerSec": 10
    },
    "diskUtilization": {
      "targetReadOpsPerSec": 2
    },
    "requestUtilization": {
      "targetRequestCountPerSec": 3
    }
  }

  Args:
    automatic_scaling: Result of converting automatic_scaling according to
      schema.
  Returns:
    AutomaticScaling which has moved network/disk utilization related fields to
    submessage.
  """

    def MoveFieldsTo(field_names, target_field_name):
        target = {}
        for field_name in field_names:
            if field_name in automatic_scaling:
                target[field_name] = automatic_scaling[field_name]
                del automatic_scaling[field_name]
        if target:
            automatic_scaling[target_field_name] = target
    MoveFieldsTo(_REQUEST_UTILIZATION_SCALING_FIELDS, 'requestUtilization')
    MoveFieldsTo(_DISK_UTILIZATION_SCALING_FIELDS, 'diskUtilization')
    MoveFieldsTo(_NETWORK_UTILIZATION_SCALING_FIELDS, 'networkUtilization')
    MoveFieldsTo(_STANDARD_SCHEDULER_SETTINGS, 'standardSchedulerSettings')
    return automatic_scaling