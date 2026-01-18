from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigdeliveryProjectsLocationsFleetPackagesRolloutsResumeRequest(_messages.Message):
    """A ConfigdeliveryProjectsLocationsFleetPackagesRolloutsResumeRequest
  object.

  Fields:
    name: Required. Name of the Rollout.
    resumeRolloutRequest: A ResumeRolloutRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    resumeRolloutRequest = _messages.MessageField('ResumeRolloutRequest', 2)