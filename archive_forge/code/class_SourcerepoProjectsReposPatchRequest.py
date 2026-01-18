from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposPatchRequest(_messages.Message):
    """A SourcerepoProjectsReposPatchRequest object.

  Fields:
    name: The name of the requested repository. Values are of the form
      `projects//repos/`.
    updateRepoRequest: A UpdateRepoRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateRepoRequest = _messages.MessageField('UpdateRepoRequest', 2)