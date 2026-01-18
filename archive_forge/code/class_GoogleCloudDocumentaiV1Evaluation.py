from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1Evaluation(_messages.Message):
    """An evaluation of a ProcessorVersion's performance.

  Messages:
    EntityMetricsValue: Metrics across confidence levels, for different
      entities.

  Fields:
    allEntitiesMetrics: Metrics for all the entities in aggregate.
    createTime: The time that the evaluation was created.
    documentCounters: Counters for the documents used in the evaluation.
    entityMetrics: Metrics across confidence levels, for different entities.
    kmsKeyName: The KMS key name used for encryption.
    kmsKeyVersionName: The KMS key version with which data is encrypted.
    name: The resource name of the evaluation. Format: `projects/{project}/loc
      ations/{location}/processors/{processor}/processorVersions/{processor_ve
      rsion}/evaluations/{evaluation}`
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EntityMetricsValue(_messages.Message):
        """Metrics across confidence levels, for different entities.

    Messages:
      AdditionalProperty: An additional property for a EntityMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type EntityMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EntityMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allEntitiesMetrics = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationMultiConfidenceMetrics', 1)
    createTime = _messages.StringField(2)
    documentCounters = _messages.MessageField('GoogleCloudDocumentaiV1EvaluationCounters', 3)
    entityMetrics = _messages.MessageField('EntityMetricsValue', 4)
    kmsKeyName = _messages.StringField(5)
    kmsKeyVersionName = _messages.StringField(6)
    name = _messages.StringField(7)