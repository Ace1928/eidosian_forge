from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsRejectRequest(_messages.Message):
    """A
  TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsRejectRequest
  object.

  Fields:
    name: Required. The name of the blueprint being rejected.
    rejectBlueprintRequest: A RejectBlueprintRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    rejectBlueprintRequest = _messages.MessageField('RejectBlueprintRequest', 2)