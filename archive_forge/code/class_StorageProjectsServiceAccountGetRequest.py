from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageProjectsServiceAccountGetRequest(_messages.Message):
    """A StorageProjectsServiceAccountGetRequest object.

  Fields:
    projectId: Project ID
    userProject: The project to be billed for this request.
  """
    projectId = _messages.StringField(1, required=True)
    userProject = _messages.StringField(2)