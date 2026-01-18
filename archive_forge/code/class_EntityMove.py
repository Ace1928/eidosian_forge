from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityMove(_messages.Message):
    """Options to configure rule type EntityMove. The rule is used to move an
  entity to a new schema. The rule filter field can refer to one or more
  entities. The rule scope can be one of: Table, Column, Constraint, Index,
  View, Function, Stored Procedure, Materialized View, Sequence, UDT

  Fields:
    newSchema: Required. The new schema
  """
    newSchema = _messages.StringField(1)