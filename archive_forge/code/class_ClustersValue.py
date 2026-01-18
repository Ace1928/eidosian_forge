from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ClustersValue(_messages.Message):
    """Required. The clusters to be created within the instance, mapped by
    desired cluster ID, e.g., just `mycluster` rather than
    `projects/myproject/instances/myinstance/clusters/mycluster`. Fields
    marked `OutputOnly` must be left blank.

    Messages:
      AdditionalProperty: An additional property for a ClustersValue object.

    Fields:
      additionalProperties: Additional properties of type ClustersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ClustersValue object.

      Fields:
        key: Name of the additional property.
        value: A Cluster attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Cluster', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)