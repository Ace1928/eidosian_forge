from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigdeliveryProjectsLocationsFleetPackagesRolloutsSuspendRequest(_messages.Message):
    """A ConfigdeliveryProjectsLocationsFleetPackagesRolloutsSuspendRequest
  object.

  Fields:
    name: Required. Name of the Rollout.
    suspendRolloutRequest: A SuspendRolloutRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    suspendRolloutRequest = _messages.MessageField('SuspendRolloutRequest', 2)