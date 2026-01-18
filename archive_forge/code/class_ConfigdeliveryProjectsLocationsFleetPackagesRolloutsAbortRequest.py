from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigdeliveryProjectsLocationsFleetPackagesRolloutsAbortRequest(_messages.Message):
    """A ConfigdeliveryProjectsLocationsFleetPackagesRolloutsAbortRequest
  object.

  Fields:
    abortRolloutRequest: A AbortRolloutRequest resource to be passed as the
      request body.
    name: Required. Name of the Rollout.
  """
    abortRolloutRequest = _messages.MessageField('AbortRolloutRequest', 1)
    name = _messages.StringField(2, required=True)