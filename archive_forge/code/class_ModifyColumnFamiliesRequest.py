from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModifyColumnFamiliesRequest(_messages.Message):
    """Request message for
  google.bigtable.admin.v2.BigtableTableAdmin.ModifyColumnFamilies

  Fields:
    ignoreWarnings: Optional. If true, ignore safety checks when modifying the
      column families.
    modifications: Required. Modifications to be atomically applied to the
      specified table's families. Entries are applied in order, meaning that
      earlier modifications can be masked by later ones (in the case of
      repeated updates to the same family, for example).
  """
    ignoreWarnings = _messages.BooleanField(1)
    modifications = _messages.MessageField('Modification', 2, repeated=True)