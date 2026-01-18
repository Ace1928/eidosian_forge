from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsProposeRequest(_messages.Message):
    """A TelcoautomationProjectsLocationsOrchestrationClustersBlueprintsPropose
  Request object.

  Fields:
    name: Required. The name of the blueprint being proposed.
    proposeBlueprintRequest: A ProposeBlueprintRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    proposeBlueprintRequest = _messages.MessageField('ProposeBlueprintRequest', 2)