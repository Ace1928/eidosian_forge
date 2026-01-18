from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsGetRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsGetRequest object.

  Enums:
    ViewValueValuesEnum: Optional. The level of details of the migrating VM.

  Fields:
    name: Required. The name of the MigratingVm.
    view: Optional. The level of details of the migrating VM.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The level of details of the migrating VM.

    Values:
      MIGRATING_VM_VIEW_UNSPECIFIED: View is unspecified. The API will
        fallback to the default value.
      MIGRATING_VM_VIEW_BASIC: Get the migrating VM basic details. The basic
        details do not include the recent clone jobs and recent cutover jobs
        lists.
      MIGRATING_VM_VIEW_FULL: Include everything.
    """
        MIGRATING_VM_VIEW_UNSPECIFIED = 0
        MIGRATING_VM_VIEW_BASIC = 1
        MIGRATING_VM_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)