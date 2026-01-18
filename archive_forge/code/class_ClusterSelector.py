from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterSelector(_messages.Message):
    """A selector that chooses target cluster for jobs based on metadata.

  Messages:
    ClusterLabelsValue: Required. The cluster labels. Cluster must have all
      labels to match.

  Fields:
    clusterLabels: Required. The cluster labels. Cluster must have all labels
      to match.
    zone: Optional. The zone where workflow process executes. This parameter
      does not affect the selection of the cluster.If unspecified, the zone of
      the first cluster matching the selector is used.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ClusterLabelsValue(_messages.Message):
        """Required. The cluster labels. Cluster must have all labels to match.

    Messages:
      AdditionalProperty: An additional property for a ClusterLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ClusterLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ClusterLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    clusterLabels = _messages.MessageField('ClusterLabelsValue', 1)
    zone = _messages.StringField(2)