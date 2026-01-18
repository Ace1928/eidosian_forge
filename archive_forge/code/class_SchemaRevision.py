from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaRevision(_messages.Message):
    """A revision of application schema.

  Fields:
    createTime: Output only. [Output only] Create time stamp.
    name: Identifier. The relative resource name of the schema revision, in
      the format: ``` projects/{project}/locations/{location}/services/{servic
      e}/schemas/{schema}/revisions/{revision} ```
    schema: Output only. The snapshot of the application schema.
    uid: Output only. System-assigned, unique identifier.
  """
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    schema = _messages.MessageField('Schema', 3)
    uid = _messages.StringField(4)