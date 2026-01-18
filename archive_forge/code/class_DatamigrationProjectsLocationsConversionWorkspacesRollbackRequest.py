from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesRollbackRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesRollbackRequest
  object.

  Fields:
    name: Required. Name of the conversion workspace resource to roll back to.
    rollbackConversionWorkspaceRequest: A RollbackConversionWorkspaceRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackConversionWorkspaceRequest = _messages.MessageField('RollbackConversionWorkspaceRequest', 2)