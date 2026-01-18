from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasProjectsLocationsQuotaPreferencesCreateRequest(_messages.Message):
    """A CloudquotasProjectsLocationsQuotaPreferencesCreateRequest object.

  Enums:
    IgnoreSafetyChecksValueValuesEnum: The list of quota safety checks to be
      ignored.

  Fields:
    ignoreSafetyChecks: The list of quota safety checks to be ignored.
    parent: Required. Value for parent. Example:
      `projects/123/locations/global`
    quotaPreference: A QuotaPreference resource to be passed as the request
      body.
    quotaPreferenceId: Optional. Id of the requesting object, must be unique
      under its parent. If client does not set this field, the service will
      generate one.
  """

    class IgnoreSafetyChecksValueValuesEnum(_messages.Enum):
        """The list of quota safety checks to be ignored.

    Values:
      QUOTA_SAFETY_CHECK_UNSPECIFIED: Unspecified quota safety check.
      QUOTA_DECREASE_BELOW_USAGE: Validates that a quota mutation would not
        cause the consumer's effective limit to be lower than the consumer's
        quota usage.
      QUOTA_DECREASE_PERCENTAGE_TOO_HIGH: Validates that a quota mutation
        would not cause the consumer's effective limit to decrease by more
        than 10 percent.
    """
        QUOTA_SAFETY_CHECK_UNSPECIFIED = 0
        QUOTA_DECREASE_BELOW_USAGE = 1
        QUOTA_DECREASE_PERCENTAGE_TOO_HIGH = 2
    ignoreSafetyChecks = _messages.EnumField('IgnoreSafetyChecksValueValuesEnum', 1, repeated=True)
    parent = _messages.StringField(2, required=True)
    quotaPreference = _messages.MessageField('QuotaPreference', 3)
    quotaPreferenceId = _messages.StringField(4)