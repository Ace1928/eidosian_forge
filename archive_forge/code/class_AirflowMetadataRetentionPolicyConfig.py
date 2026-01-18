from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AirflowMetadataRetentionPolicyConfig(_messages.Message):
    """The policy for airflow metadata database retention.

  Enums:
    RetentionModeValueValuesEnum: Optional. Retention can be either enabled or
      disabled.

  Fields:
    retentionDays: Optional. How many days data should be retained for.
    retentionMode: Optional. Retention can be either enabled or disabled.
  """

    class RetentionModeValueValuesEnum(_messages.Enum):
        """Optional. Retention can be either enabled or disabled.

    Values:
      RETENTION_MODE_UNSPECIFIED: Default mode doesn't change environment
        parameters.
      RETENTION_MODE_ENABLED: Retention policy is enabled.
      RETENTION_MODE_DISABLED: Retention policy is disabled.
    """
        RETENTION_MODE_UNSPECIFIED = 0
        RETENTION_MODE_ENABLED = 1
        RETENTION_MODE_DISABLED = 2
    retentionDays = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    retentionMode = _messages.EnumField('RetentionModeValueValuesEnum', 2)