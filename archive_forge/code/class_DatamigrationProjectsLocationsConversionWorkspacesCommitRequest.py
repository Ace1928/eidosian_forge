from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesCommitRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesCommitRequest
  object.

  Fields:
    commitConversionWorkspaceRequest: A CommitConversionWorkspaceRequest
      resource to be passed as the request body.
    name: Required. Name of the conversion workspace resource to commit.
  """
    commitConversionWorkspaceRequest = _messages.MessageField('CommitConversionWorkspaceRequest', 1)
    name = _messages.StringField(2, required=True)