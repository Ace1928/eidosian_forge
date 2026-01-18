from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsUpdateConfigRequest(_messages.Message):
    """A SourcerepoProjectsUpdateConfigRequest object.

  Fields:
    name: The name of the requested project. Values are of the form
      `projects/`.
    updateProjectConfigRequest: A UpdateProjectConfigRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    updateProjectConfigRequest = _messages.MessageField('UpdateProjectConfigRequest', 2)