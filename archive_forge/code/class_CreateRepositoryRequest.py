from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateRepositoryRequest(_messages.Message):
    """Message for creating a Repository.

  Fields:
    parent: Required. The connection to contain the repository. If the request
      is part of a BatchCreateRepositoriesRequest, this field should be empty
      or match the parent specified there.
    repository: Required. The repository to create.
    repositoryId: Required. The ID to use for the repository, which will
      become the final component of the repository's resource name. This ID
      should be unique in the connection. Allows alphanumeric characters and
      any of -._~%!$&'()*+,;=@.
  """
    parent = _messages.StringField(1)
    repository = _messages.MessageField('Repository', 2)
    repositoryId = _messages.StringField(3)