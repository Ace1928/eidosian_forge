from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsRolloutsResumeRequest(_messages.Message):
    """A GkehubProjectsLocationsRolloutsResumeRequest object.

  Fields:
    name: Required. The name of the rollout to resume.
      projects/{project}/locations/{location}/rollouts/{rollout}
    resumeRolloutRequest: A ResumeRolloutRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resumeRolloutRequest = _messages.MessageField('ResumeRolloutRequest', 2)