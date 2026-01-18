from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisioningIssue(_messages.Message):
    """Information about issues with provisioning a Managed Certificate.

  Enums:
    ReasonValueValuesEnum: Output only. Reason for provisioning failures.

  Fields:
    details: Output only. Human readable explanation about the issue. Provided
      to help address the configuration issues. Not guaranteed to be stable.
      For programmatic access use Reason enum.
    reason: Output only. Reason for provisioning failures.
  """

    class ReasonValueValuesEnum(_messages.Enum):
        """Output only. Reason for provisioning failures.

    Values:
      REASON_UNSPECIFIED: <no description>
      AUTHORIZATION_ISSUE: Certificate provisioning failed due to an issue
        with one or more of the domains on the certificate. For details of
        which domains failed, consult the `authorization_attempt_info` field.
      RATE_LIMITED: Exceeded Certificate Authority quotas or internal rate
        limits of the system. Provisioning may take longer to complete.
    """
        REASON_UNSPECIFIED = 0
        AUTHORIZATION_ISSUE = 1
        RATE_LIMITED = 2
    details = _messages.StringField(1)
    reason = _messages.EnumField('ReasonValueValuesEnum', 2)