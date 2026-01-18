from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorRevision(_messages.Message):
    """A revision of a connector.

  Fields:
    connector: Output only. The snapshot of the connector.
    createTime: Output only. [Output only] Create time stamp.
    name: Identifier. The relative resource name of the connector revision, in
      the format: ``` projects/{project}/locations/{location}/services/{servic
      e}/connectors/{connector}/revisions/{revision} ```
    uid: Output only. System-assigned, unique identifier.
  """
    connector = _messages.MessageField('Connector', 1)
    createTime = _messages.StringField(2)
    name = _messages.StringField(3)
    uid = _messages.StringField(4)