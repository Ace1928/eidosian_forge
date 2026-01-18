from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsInstallationsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsInstallationsPatchRequest object.

  Fields:
    installation: A Installation resource to be passed as the request body.
    installationId: Unique identifier of the GitHub installation. Deprecated.
      Should set installation.id
    name: The `Installation` name with format:
      `projects/{project}/locations/{location}/installations/{installation}`,
      where {installation} is GitHub installation ID created by GitHub.
    name1: The name of the `GitHubInstallation` to update. Format:
      `projects/{project}/locations/{location}/installations/{installation}`
    projectId: ID of the project.
    updateMask: Update mask for the Installation resource. If this is set, the
      server will only update the fields specified in the field mask.
      Otherwise, a full update of the resource will be performed.
  """
    installation = _messages.MessageField('Installation', 1)
    installationId = _messages.IntegerField(2)
    name = _messages.StringField(3, required=True)
    name1 = _messages.StringField(4)
    projectId = _messages.StringField(5)
    updateMask = _messages.StringField(6)