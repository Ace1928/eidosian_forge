from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SingleEntityRename(_messages.Message):
    """Options to configure rule type SingleEntityRename. The rule is used to
  rename an entity. The rule filter field can refer to only one entity. The
  rule scope can be one of: Database, Schema, Table, Column, Constraint,
  Index, View, Function, Stored Procedure, Materialized View, Sequence, UDT,
  Synonym

  Fields:
    newName: Required. The new name of the destination entity
  """
    newName = _messages.StringField(1)