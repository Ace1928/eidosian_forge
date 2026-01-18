from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsTriggersRunRequest(_messages.Message):
    """A CloudbuildProjectsLocationsTriggersRunRequest object.

  Fields:
    name: The name of the `Trigger` to run. Format:
      `projects/{project}/locations/{location}/triggers/{trigger}`
    runBuildTriggerRequest: A RunBuildTriggerRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    runBuildTriggerRequest = _messages.MessageField('RunBuildTriggerRequest', 2)