from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsTriggersRunRequest(_messages.Message):
    """A CloudbuildProjectsTriggersRunRequest object.

  Fields:
    name: The name of the `Trigger` to run. Format:
      `projects/{project}/locations/{location}/triggers/{trigger}`
    projectId: Required. ID of the project.
    repoSource: A RepoSource resource to be passed as the request body.
    triggerId: Required. ID of the trigger.
  """
    name = _messages.StringField(1)
    projectId = _messages.StringField(2, required=True)
    repoSource = _messages.MessageField('RepoSource', 3)
    triggerId = _messages.StringField(4, required=True)