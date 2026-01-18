from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterOperationMetadata(_messages.Message):
    """Metadata describing the operation.

  Messages:
    LabelsValue: Output only. Labels associated with the operation

  Fields:
    childOperationIds: Output only. Child operation ids
    clusterName: Output only. Name of the cluster for the operation.
    clusterUuid: Output only. Cluster UUID for the operation.
    description: Output only. Short description of operation.
    labels: Output only. Labels associated with the operation
    operationType: Output only. The operation type.
    status: Output only. Current operation status.
    statusHistory: Output only. The previous operation status.
    warnings: Output only. Errors encountered during operation execution.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Output only. Labels associated with the operation

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    childOperationIds = _messages.StringField(1, repeated=True)
    clusterName = _messages.StringField(2)
    clusterUuid = _messages.StringField(3)
    description = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    operationType = _messages.StringField(6)
    status = _messages.MessageField('ClusterOperationStatus', 7)
    statusHistory = _messages.MessageField('ClusterOperationStatus', 8, repeated=True)
    warnings = _messages.StringField(9, repeated=True)