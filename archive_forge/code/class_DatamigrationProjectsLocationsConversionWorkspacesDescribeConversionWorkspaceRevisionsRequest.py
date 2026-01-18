from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesDescribeConversionWorkspaceRevisionsRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesDescribeConversionWo
  rkspaceRevisionsRequest object.

  Fields:
    commitId: Optional. Optional filter to request a specific commit ID.
    conversionWorkspace: Required. Name of the conversion workspace resource
      whose revisions are listed. Must be in the form of: projects/{project}/l
      ocations/{location}/conversionWorkspaces/{conversion_workspace}.
  """
    commitId = _messages.StringField(1)
    conversionWorkspace = _messages.StringField(2, required=True)