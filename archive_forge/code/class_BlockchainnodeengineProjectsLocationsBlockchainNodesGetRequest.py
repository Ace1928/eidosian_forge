from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlockchainnodeengineProjectsLocationsBlockchainNodesGetRequest(_messages.Message):
    """A BlockchainnodeengineProjectsLocationsBlockchainNodesGetRequest object.

  Fields:
    name: Required. The fully qualified name of the blockchain node to fetch.
      e.g. `projects/my-project/locations/us-central1/blockchainNodes/my-
      node`.
  """
    name = _messages.StringField(1, required=True)