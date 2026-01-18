from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsStartRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsStartRequest object.

  Fields:
    name: Required. Name of the workstation to start.
    startWorkstationRequest: A StartWorkstationRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    startWorkstationRequest = _messages.MessageField('StartWorkstationRequest', 2)