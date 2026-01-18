from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStopRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsStopRequest object.

  Fields:
    name: Required. Name of the workstation to stop.
    stopWorkstationRequest: A StopWorkstationRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    stopWorkstationRequest = _messages.MessageField('StopWorkstationRequest', 2)