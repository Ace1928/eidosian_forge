from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsGetRequest(_messages.Message):
    """A BlockchainvalidatormanagerProjectsLocationsBlockchainValidatorConfigsG
  etRequest object.

  Fields:
    name: Required. The resource name of the validator config. This is derived
      from the public key, however it is sensitive due to the inclusion of the
      project ID in the resource name. e.g. `projects/my-project/locations/us-
      central1/blockchainValidatorConfigs/0xa75dbe920352d3e91d06bd8cfe8eb67867
      7127f8748854a7a8894e3c121b63169259448a4b63e5cfb992da412ac91c30`.
  """
    name = _messages.StringField(1, required=True)