from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2FulfillmentFeature(_messages.Message):
    """Whether fulfillment is enabled for the specific feature.

  Enums:
    TypeValueValuesEnum: The type of the feature that enabled for fulfillment.

  Fields:
    type: The type of the feature that enabled for fulfillment.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of the feature that enabled for fulfillment.

    Values:
      TYPE_UNSPECIFIED: Feature type not specified.
      SMALLTALK: Fulfillment is enabled for SmallTalk.
    """
        TYPE_UNSPECIFIED = 0
        SMALLTALK = 1
    type = _messages.EnumField('TypeValueValuesEnum', 1)