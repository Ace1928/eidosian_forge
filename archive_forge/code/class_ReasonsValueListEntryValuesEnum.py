from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReasonsValueListEntryValuesEnum(_messages.Enum):
    """ReasonsValueListEntryValuesEnum enum type.

    Values:
      CLASSIFICATION_REASON_UNSPECIFIED: Default unspecified type.
      AUTOMATION: Interactions matched the behavior of an automated agent.
      UNEXPECTED_ENVIRONMENT: The event originated from an illegitimate
        environment.
      TOO_MUCH_TRAFFIC: Traffic volume from the event source is higher than
        normal.
      UNEXPECTED_USAGE_PATTERNS: Interactions with the site were significantly
        different than expected patterns.
      LOW_CONFIDENCE_SCORE: Too little traffic has been received from this
        site thus far to generate quality risk analysis.
      SUSPECTED_CARDING: The request matches behavioral characteristics of a
        carding attack.
      SUSPECTED_CHARGEBACK: The request matches behavioral characteristics of
        chargebacks for fraud.
    """
    CLASSIFICATION_REASON_UNSPECIFIED = 0
    AUTOMATION = 1
    UNEXPECTED_ENVIRONMENT = 2
    TOO_MUCH_TRAFFIC = 3
    UNEXPECTED_USAGE_PATTERNS = 4
    LOW_CONFIDENCE_SCORE = 5
    SUSPECTED_CARDING = 6
    SUSPECTED_CHARGEBACK = 7