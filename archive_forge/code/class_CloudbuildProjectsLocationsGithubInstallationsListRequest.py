from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsGithubInstallationsListRequest(_messages.Message):
    """A CloudbuildProjectsLocationsGithubInstallationsListRequest object.

  Fields:
    parent: The parent resource where github installations for project will be
      listed. Format: `projects/{project}/locations/{location}`
    projectId: Project id
  """
    parent = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2)