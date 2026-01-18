from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookerProjectsLocationsInstancesRestartRequest(_messages.Message):
    """A LookerProjectsLocationsInstancesRestartRequest object.

  Fields:
    name: Required. Format:
      `projects/{project}/locations/{location}/instances/{instance}`.
    restartInstanceRequest: A RestartInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    restartInstanceRequest = _messages.MessageField('RestartInstanceRequest', 2)