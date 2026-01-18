from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SuppressionInfo(_messages.Message):
    """Information about entries that were omitted from the session.

  Enums:
    ReasonValueValuesEnum: The reason that entries were omitted from the
      session.

  Fields:
    reason: The reason that entries were omitted from the session.
    suppressedCount: A lower bound on the count of entries omitted due to
      reason.
  """

    class ReasonValueValuesEnum(_messages.Enum):
        """The reason that entries were omitted from the session.

    Values:
      REASON_UNSPECIFIED: Unexpected default.
      RATE_LIMIT: Indicates suppression occurred due to relevant entries being
        received in excess of rate limits. For quotas and limits, see Logging
        API quotas and limits (https://cloud.google.com/logging/quotas#api-
        limits).
      NOT_CONSUMED: Indicates suppression occurred due to the client not
        consuming responses quickly enough.
    """
        REASON_UNSPECIFIED = 0
        RATE_LIMIT = 1
        NOT_CONSUMED = 2
    reason = _messages.EnumField('ReasonValueValuesEnum', 1)
    suppressedCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)