from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasProjectsLocationsQuotaPreferencesPatchRequest(_messages.Message):
    """A CloudquotasProjectsLocationsQuotaPreferencesPatchRequest object.

  Enums:
    IgnoreSafetyChecksValueValuesEnum: The list of quota safety checks to be
      ignored.

  Fields:
    allowMissing: Optional. If set to true, and the quota preference is not
      found, a new one will be created. In this situation, `update_mask` is
      ignored.
    ignoreSafetyChecks: The list of quota safety checks to be ignored.
    name: Required except in the CREATE requests. The resource name of the
      quota preference. The ID component following "locations/" must be
      "global". Example: `projects/123/locations/global/quotaPreferences/my-
      config-for-us-east1`
    quotaPreference: A QuotaPreference resource to be passed as the request
      body.
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the QuotaPreference resource by the update. The fields
      specified in the update_mask are relative to the resource, not the full
      request. A field will be overwritten if it is in the mask. If the user
      does not provide a mask then all fields will be overwritten.
    validateOnly: Optional. If set to true, validate the request, but do not
      actually update. Note that a request being valid does not mean that the
      request is guaranteed to be fulfilled.
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
    allowMissing = _messages.BooleanField(1)
    ignoreSafetyChecks = _messages.EnumField('IgnoreSafetyChecksValueValuesEnum', 2, repeated=True)
    name = _messages.StringField(3, required=True)
    quotaPreference = _messages.MessageField('QuotaPreference', 4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)