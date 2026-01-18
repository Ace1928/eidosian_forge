from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleColumn(_messages.Message):
    """Oracle Column.

  Fields:
    column: Column name.
    dataType: The Oracle data type.
    encoding: Column encoding.
    length: Column length.
    nullable: Whether or not the column can accept a null value.
    ordinalPosition: The ordinal position of the column in the table.
    precision: Column precision.
    primaryKey: Whether or not the column represents a primary key.
    scale: Column scale.
  """
    column = _messages.StringField(1)
    dataType = _messages.StringField(2)
    encoding = _messages.StringField(3)
    length = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    nullable = _messages.BooleanField(5)
    ordinalPosition = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    precision = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    primaryKey = _messages.BooleanField(8)
    scale = _messages.IntegerField(9, variant=_messages.Variant.INT32)