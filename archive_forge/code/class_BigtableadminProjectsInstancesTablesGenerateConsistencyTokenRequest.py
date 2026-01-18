from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesTablesGenerateConsistencyTokenRequest(_messages.Message):
    """A BigtableadminProjectsInstancesTablesGenerateConsistencyTokenRequest
  object.

  Fields:
    generateConsistencyTokenRequest: A GenerateConsistencyTokenRequest
      resource to be passed as the request body.
    name: Required. The unique name of the Table for which to create a
      consistency token. Values are of the form
      `projects/{project}/instances/{instance}/tables/{table}`.
  """
    generateConsistencyTokenRequest = _messages.MessageField('GenerateConsistencyTokenRequest', 1)
    name = _messages.StringField(2, required=True)