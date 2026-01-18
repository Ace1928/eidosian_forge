from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1AccountDefenderAssessment(_messages.Message):
    """Account defender risk assessment.

  Enums:
    LabelsValueListEntryValuesEnum:

  Fields:
    labels: Output only. Labels for this request.
  """

    class LabelsValueListEntryValuesEnum(_messages.Enum):
        """LabelsValueListEntryValuesEnum enum type.

    Values:
      ACCOUNT_DEFENDER_LABEL_UNSPECIFIED: Default unspecified type.
      PROFILE_MATCH: The request matches a known good profile for the user.
      SUSPICIOUS_LOGIN_ACTIVITY: The request is potentially a suspicious login
        event and must be further verified either through multi-factor
        authentication or another system.
      SUSPICIOUS_ACCOUNT_CREATION: The request matched a profile that
        previously had suspicious account creation behavior. This can mean
        that this is a fake account.
      RELATED_ACCOUNTS_NUMBER_HIGH: The account in the request has a high
        number of related accounts. It does not necessarily imply that the
        account is bad but can require further investigation.
    """
        ACCOUNT_DEFENDER_LABEL_UNSPECIFIED = 0
        PROFILE_MATCH = 1
        SUSPICIOUS_LOGIN_ACTIVITY = 2
        SUSPICIOUS_ACCOUNT_CREATION = 3
        RELATED_ACCOUNTS_NUMBER_HIGH = 4
    labels = _messages.EnumField('LabelsValueListEntryValuesEnum', 1, repeated=True)