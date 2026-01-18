from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetsPresentCondition(_messages.Message):
    """`TargetsPresentCondition` contains information on any Targets referenced
  in the Delivery Pipeline that do not actually exist.

  Fields:
    missingTargets: The list of Target names that do not exist. For example,
      `projects/{project_id}/locations/{location_name}/targets/{target_name}`.
    status: True if there aren't any missing Targets.
    updateTime: Last time the condition was updated.
  """
    missingTargets = _messages.StringField(1, repeated=True)
    status = _messages.BooleanField(2)
    updateTime = _messages.StringField(3)