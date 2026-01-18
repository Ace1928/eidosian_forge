from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWorkstationsGenerateAccessTokenRequest(_messages.Message):
    """A WorkstationsProjectsLocationsWorkstationClustersWorkstationConfigsWork
  stationsGenerateAccessTokenRequest object.

  Fields:
    generateAccessTokenRequest: A GenerateAccessTokenRequest resource to be
      passed as the request body.
    workstation: Required. Name of the workstation for which the access token
      should be generated.
  """
    generateAccessTokenRequest = _messages.MessageField('GenerateAccessTokenRequest', 1)
    workstation = _messages.StringField(2, required=True)