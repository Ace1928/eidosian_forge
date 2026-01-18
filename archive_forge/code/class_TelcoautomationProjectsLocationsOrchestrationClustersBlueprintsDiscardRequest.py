from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDiscardRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsDiscard
  Request object.

  Fields:
    discardBlueprintChangesRequest: A DiscardBlueprintChangesRequest resource
      to be passed as the request body.
    name: Required. The name of the blueprint of which changes are being
      discarded.
  """
    discardBlueprintChangesRequest = _messages.MessageField('DiscardBlueprintChangesRequest', 1)
    name = _messages.StringField(2, required=True)