from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesUndeleteRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesUndeleteRequest object.

  Fields:
    name: Required. The unique name of the table to be restored. Values are of
      the form `projects/{project}/instances/{instance}/tables/{table}`.
    undeleteTableRequest: A UndeleteTableRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteTableRequest = _messages.MessageField('UndeleteTableRequest', 2)