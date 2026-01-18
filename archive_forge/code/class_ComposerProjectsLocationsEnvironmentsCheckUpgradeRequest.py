from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsCheckUpgradeRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsCheckUpgradeRequest object.

  Fields:
    checkUpgradeRequest: A CheckUpgradeRequest resource to be passed as the
      request body.
    environment: The resource name of the environment to check upgrade for, in
      the form: "projects/{projectId}/locations/{locationId}/environments/{env
      ironmentId}"
  """
    checkUpgradeRequest = _messages.MessageField('CheckUpgradeRequest', 1)
    environment = _messages.StringField(2, required=True)