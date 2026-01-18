from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Modification(_messages.Message):
    """A create, update, or delete of a particular column family.

  Fields:
    create: Create a new column family with the specified schema, or fail if
      one already exists with the given ID.
    drop: Drop (delete) the column family with the given ID, or fail if no
      such family exists.
    id: The ID of the column family to be modified.
    update: Update an existing column family to the specified schema, or fail
      if no column family exists with the given ID.
    updateMask: Optional. A mask specifying which fields (e.g. `gc_rule`) in
      the `update` mod should be updated, ignored for other modification
      types. If unset or empty, we treat it as updating `gc_rule` to be
      backward compatible.
  """
    create = _messages.MessageField('ColumnFamily', 1)
    drop = _messages.BooleanField(2)
    id = _messages.StringField(3)
    update = _messages.MessageField('ColumnFamily', 4)
    updateMask = _messages.StringField(5)