from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsInstallationsCreateRequest(_messages.Message):
    """A CloudbuildProjectsInstallationsCreateRequest object.

  Fields:
    installation: A Installation resource to be passed as the request body.
    parent: The parent resource where this github installation will be
      created. Format: `projects/{project}/locations/{location}`
    projectId: ID of the project.
    userOauthCode: GitHub user code. If a GitHub credential is already
      associated with the user this can be omitted, else the code is used to
      exchange and store an OAuth token.
  """
    installation = _messages.MessageField('Installation', 1)
    parent = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)
    userOauthCode = _messages.StringField(4)