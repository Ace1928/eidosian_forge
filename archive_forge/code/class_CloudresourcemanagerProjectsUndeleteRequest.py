from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudresourcemanagerProjectsUndeleteRequest(_messages.Message):
    """A CloudresourcemanagerProjectsUndeleteRequest object.

  Fields:
    projectId: Required. The project ID (for example, `foo-bar-123`).
    undeleteProjectRequest: A UndeleteProjectRequest resource to be passed as
      the request body.
  """
    projectId = _messages.StringField(1, required=True)
    undeleteProjectRequest = _messages.MessageField('UndeleteProjectRequest', 2)