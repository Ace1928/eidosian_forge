from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementCluster(_messages.Message):
    """Management cluster configuration.

  Messages:
    NodeTypeConfigsValue: Required. The map of cluster node types in this
      cluster, where the key is canonical identifier of the node type
      (corresponds to the `NodeType`).

  Fields:
    autoscalingSettings: Optional. Configuration of the autoscaling applied to
      this cluster.
    clusterId: Required. The user-provided identifier of the new `Cluster`.
      The identifier must meet the following requirements: * Only contains
      1-63 alphanumeric characters and hyphens * Begins with an alphabetical
      character * Ends with a non-hyphen character * Not formatted as a UUID *
      Complies with [RFC 1034](https://datatracker.ietf.org/doc/html/rfc1034)
      (section 3.5)
    nodeCount: Optional. Deprecated: Number of nodes in this cluster.
    nodeCustomCoreCount: Optional. Deprecated: Customized number of cores
      available to each node of the cluster. This number must always be one of
      `nodeType.availableCustomCoreCounts`. If zero is provided max value from
      `nodeType.availableCustomCoreCounts` will be used.
    nodeTypeConfigs: Required. The map of cluster node types in this cluster,
      where the key is canonical identifier of the node type (corresponds to
      the `NodeType`).
    nodeTypeId: Optional. Deprecated: The canonical identifier of node types
      (`NodeType`) in this cluster. For example: standard-72.
    stretchedClusterConfig: Optional. Configuration of a stretched cluster.
      Required for STRETCHED private clouds.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NodeTypeConfigsValue(_messages.Message):
        """Required. The map of cluster node types in this cluster, where the key
    is canonical identifier of the node type (corresponds to the `NodeType`).

    Messages:
      AdditionalProperty: An additional property for a NodeTypeConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type NodeTypeConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NodeTypeConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A NodeTypeConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('NodeTypeConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    autoscalingSettings = _messages.MessageField('AutoscalingSettings', 1)
    clusterId = _messages.StringField(2)
    nodeCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    nodeCustomCoreCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    nodeTypeConfigs = _messages.MessageField('NodeTypeConfigsValue', 5)
    nodeTypeId = _messages.StringField(6)
    stretchedClusterConfig = _messages.MessageField('StretchedClusterConfig', 7)