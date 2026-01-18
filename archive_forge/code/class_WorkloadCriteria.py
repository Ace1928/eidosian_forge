from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadCriteria(_messages.Message):
    """Criteria to apply to identify assets belonging to this workload. Used to
  auto-populate the assets field.

  Enums:
    KeyValueValuesEnum: Required. Key for criteria.

  Fields:
    key: Required. Key for criteria.
    value: Required. Criteria value to match against for the associated
      criteria key. Example: //compute.googleapis.com/projects/123/regions/us-
      west1/backendServices/bs1
  """

    class KeyValueValuesEnum(_messages.Enum):
        """Required. Key for criteria.

    Values:
      CRITERIA_KEY_UNSPECIFIED: Default. Criteria.key is unspecified.
      INSTANCE_GROUP: The criteria key is Instance Group.
      BACKEND_SERVICE: The criteria key is Backend Service.
    """
        CRITERIA_KEY_UNSPECIFIED = 0
        INSTANCE_GROUP = 1
        BACKEND_SERVICE = 2
    key = _messages.EnumField('KeyValueValuesEnum', 1)
    value = _messages.StringField(2)