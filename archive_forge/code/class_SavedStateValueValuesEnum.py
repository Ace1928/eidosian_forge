from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SavedStateValueValuesEnum(_messages.Enum):
    """For LocalSSD disks on VM Instances in STOPPED or SUSPENDED state, this
    field is set to PRESERVED if the LocalSSD data has been saved to a
    persistent location by customer request. (see the discard_local_ssd option
    on Stop/Suspend). Read-only in the api.

    Values:
      DISK_SAVED_STATE_UNSPECIFIED: *[Default]* Disk state has not been
        preserved.
      PRESERVED: Disk state has been preserved.
    """
    DISK_SAVED_STATE_UNSPECIFIED = 0
    PRESERVED = 1