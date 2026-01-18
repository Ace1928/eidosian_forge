from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudSignalsCardSignals(_messages.Message):
    """Signals describing the payment card used in this transaction.

  Enums:
    CardLabelsValueListEntryValuesEnum:

  Fields:
    cardLabels: Output only. The labels for the payment card in this
      transaction.
  """

    class CardLabelsValueListEntryValuesEnum(_messages.Enum):
        """CardLabelsValueListEntryValuesEnum enum type.

    Values:
      CARD_LABEL_UNSPECIFIED: No label specified.
      PREPAID: This card has been detected as prepaid.
      VIRTUAL: This card has been detected as virtual, such as a card number
        generated for a single transaction or merchant.
      UNEXPECTED_LOCATION: This card has been detected as being used in an
        unexpected geographic location.
    """
        CARD_LABEL_UNSPECIFIED = 0
        PREPAID = 1
        VIRTUAL = 2
        UNEXPECTED_LOCATION = 3
    cardLabels = _messages.EnumField('CardLabelsValueListEntryValuesEnum', 1, repeated=True)