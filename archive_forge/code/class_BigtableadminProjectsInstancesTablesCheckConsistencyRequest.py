from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesCheckConsistencyRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesCheckConsistencyRequest object.

  Fields:
    checkConsistencyRequest: A CheckConsistencyRequest resource to be passed
      as the request body.
    name: Required. The unique name of the Table for which to check
      replication consistency. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}`.
  """
    checkConsistencyRequest = _messages.MessageField('CheckConsistencyRequest', 1)
    name = _messages.StringField(2, required=True)