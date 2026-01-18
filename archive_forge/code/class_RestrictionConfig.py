from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestrictionConfig(_messages.Message):
    """A RestrictionConfig object.

  Enums:
    TypeValueValuesEnum: Output only. Specifies the type of dataset/table
      restriction.

  Fields:
    type: Output only. Specifies the type of dataset/table restriction.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. Specifies the type of dataset/table restriction.

    Values:
      RESTRICTION_TYPE_UNSPECIFIED: Should never be used.
      RESTRICTED_DATA_EGRESS: Restrict data egress. See [Data
        egress](/bigquery/docs/analytics-hub-introduction#data_egress) for
        more details.
    """
        RESTRICTION_TYPE_UNSPECIFIED = 0
        RESTRICTED_DATA_EGRESS = 1
    type = _messages.EnumField('TypeValueValuesEnum', 1)