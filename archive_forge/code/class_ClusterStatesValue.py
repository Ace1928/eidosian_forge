from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ClusterStatesValue(_messages.Message):
    """Output only. Map from cluster ID to per-cluster table state. If it
    could not be determined whether or not the table has data in a particular
    cluster (for example, if its zone is unavailable), then there will be an
    entry for the cluster with UNKNOWN `replication_status`. Views:
    `REPLICATION_VIEW`, `ENCRYPTION_VIEW`, `FULL`

    Messages:
      AdditionalProperty: An additional property for a ClusterStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ClusterStatesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ClusterStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A ClusterState attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ClusterState', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)