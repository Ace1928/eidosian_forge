from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGenerateRequest(_messages.Message):
    """A BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsG
  enerateRequest object.

  Fields:
    generateBlockchainValidatorConfigsRequest: A
      GenerateBlockchainValidatorConfigsRequest resource to be passed as the
      request body.
    parent: Required. The parent location to create validator configurations
      under. Format: projects/{project_number}/locations/{location}.
  """
    generateBlockchainValidatorConfigsRequest = _messages.MessageField('GenerateBlockchainValidatorConfigsRequest', 1)
    parent = _messages.StringField(2, required=True)