from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ColumnScoreValue(_messages.Message):
    """The score of each column scanned in the data scan job. The key of the
    map is the name of the column. The value is the data quality score for the
    column.The score ranges between 0, 100 (up to two decimal points).

    Messages:
      AdditionalProperty: An additional property for a ColumnScoreValue
        object.

    Fields:
      additionalProperties: Additional properties of type ColumnScoreValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ColumnScoreValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
        key = _messages.StringField(1)
        value = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)