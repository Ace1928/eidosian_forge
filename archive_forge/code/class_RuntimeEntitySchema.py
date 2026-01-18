from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeEntitySchema(_messages.Message):
    """Schema of a runtime entity.

  Fields:
    entity: Output only. Name of the entity.
    fields: Output only. List of fields in the entity.
  """
    entity = _messages.StringField(1)
    fields = _messages.MessageField('Field', 2, repeated=True)