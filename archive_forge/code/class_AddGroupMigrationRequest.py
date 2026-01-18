from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddGroupMigrationRequest(_messages.Message):
    """Request message for 'AddGroupMigration' request.

  Fields:
    migratingVm: The full path name of the MigratingVm to add.
  """
    migratingVm = _messages.StringField(1)