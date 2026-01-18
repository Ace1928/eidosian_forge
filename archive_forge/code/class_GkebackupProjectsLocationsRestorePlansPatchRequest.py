from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkebackupProjectsLocationsRestorePlansPatchRequest(_messages.Message):
    """A GkebackupProjectsLocationsRestorePlansPatchRequest object.

  Fields:
    name: Output only. The full name of the RestorePlan resource. Format:
      `projects/*/locations/*/restorePlans/*`.
    restorePlan: A RestorePlan resource to be passed as the request body.
    updateMask: Optional. This is used to specify the fields to be overwritten
      in the RestorePlan targeted for update. The values for each of these
      updated fields will be taken from the `restore_plan` provided with this
      request. Field names are relative to the root of the resource. If no
      `update_mask` is provided, all fields in `restore_plan` will be written
      to the target RestorePlan resource. Note that OUTPUT_ONLY and IMMUTABLE
      fields in `restore_plan` are ignored and are not used to update the
      target RestorePlan.
  """
    name = _messages.StringField(1, required=True)
    restorePlan = _messages.MessageField('RestorePlan', 2)
    updateMask = _messages.StringField(3)