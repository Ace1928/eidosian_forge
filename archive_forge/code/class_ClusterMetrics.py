from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterMetrics(_messages.Message):
    """Contains cluster daemon metrics, such as HDFS and YARN stats.Beta
  Feature: This report is available for testing purposes only. It may be
  changed before final release.

  Messages:
    HdfsMetricsValue: The HDFS metrics.
    SparkMetricsValue: Spark metrics.
    YarnMetricsValue: YARN metrics.

  Fields:
    hdfsMetrics: The HDFS metrics.
    sparkMetrics: Spark metrics.
    yarnMetrics: YARN metrics.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class HdfsMetricsValue(_messages.Message):
        """The HDFS metrics.

    Messages:
      AdditionalProperty: An additional property for a HdfsMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type HdfsMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a HdfsMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SparkMetricsValue(_messages.Message):
        """Spark metrics.

    Messages:
      AdditionalProperty: An additional property for a SparkMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type SparkMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SparkMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class YarnMetricsValue(_messages.Message):
        """YARN metrics.

    Messages:
      AdditionalProperty: An additional property for a YarnMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type YarnMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a YarnMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    hdfsMetrics = _messages.MessageField('HdfsMetricsValue', 1)
    sparkMetrics = _messages.MessageField('SparkMetricsValue', 2)
    yarnMetrics = _messages.MessageField('YarnMetricsValue', 3)