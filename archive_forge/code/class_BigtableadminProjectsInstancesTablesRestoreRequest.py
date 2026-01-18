from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesRestoreRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesRestoreRequest object.

  Fields:
    parent: Required. The name of the instance in which to create the restored
      table. Values are of the form `projects//instances/`.
    restoreTableRequest: A RestoreTableRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    restoreTableRequest = _messages.MessageField('RestoreTableRequest', 2)