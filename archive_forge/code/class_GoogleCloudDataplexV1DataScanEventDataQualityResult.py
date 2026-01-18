from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataScanEventDataQualityResult(_messages.Message):
    """Data quality result for data scan job.

  Messages:
    ColumnScoreValue: The score of each column scanned in the data scan job.
      The key of the map is the name of the column. The value is the data
      quality score for the column.The score ranges between 0, 100 (up to two
      decimal points).
    DimensionPassedValue: The result of each dimension for data quality
      result. The key of the map is the name of the dimension. The value is
      the bool value depicting whether the dimension result was pass or not.
    DimensionScoreValue: The score of each dimension for data quality result.
      The key of the map is the name of the dimension. The value is the data
      quality score for the dimension.The score ranges between 0, 100 (up to
      two decimal points).

  Fields:
    columnScore: The score of each column scanned in the data scan job. The
      key of the map is the name of the column. The value is the data quality
      score for the column.The score ranges between 0, 100 (up to two decimal
      points).
    dimensionPassed: The result of each dimension for data quality result. The
      key of the map is the name of the dimension. The value is the bool value
      depicting whether the dimension result was pass or not.
    dimensionScore: The score of each dimension for data quality result. The
      key of the map is the name of the dimension. The value is the data
      quality score for the dimension.The score ranges between 0, 100 (up to
      two decimal points).
    passed: Whether the data quality result was pass or not.
    rowCount: The count of rows processed in the data scan job.
    score: The table-level data quality score for the data scan job.The data
      quality score ranges between 0, 100 (up to two decimal points).
  """

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionPassedValue(_messages.Message):
        """The result of each dimension for data quality result. The key of the
    map is the name of the dimension. The value is the bool value depicting
    whether the dimension result was pass or not.

    Messages:
      AdditionalProperty: An additional property for a DimensionPassedValue
        object.

    Fields:
      additionalProperties: Additional properties of type DimensionPassedValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionPassedValue object.

      Fields:
        key: Name of the additional property.
        value: A boolean attribute.
      """
            key = _messages.StringField(1)
            value = _messages.BooleanField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DimensionScoreValue(_messages.Message):
        """The score of each dimension for data quality result. The key of the
    map is the name of the dimension. The value is the data quality score for
    the dimension.The score ranges between 0, 100 (up to two decimal points).

    Messages:
      AdditionalProperty: An additional property for a DimensionScoreValue
        object.

    Fields:
      additionalProperties: Additional properties of type DimensionScoreValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DimensionScoreValue object.

      Fields:
        key: Name of the additional property.
        value: A number attribute.
      """
            key = _messages.StringField(1)
            value = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    columnScore = _messages.MessageField('ColumnScoreValue', 1)
    dimensionPassed = _messages.MessageField('DimensionPassedValue', 2)
    dimensionScore = _messages.MessageField('DimensionScoreValue', 3)
    passed = _messages.BooleanField(4)
    rowCount = _messages.IntegerField(5)
    score = _messages.FloatField(6, variant=_messages.Variant.FLOAT)