from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresExecutionsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresExecutionsGetRequest object.

  Fields:
    name: Required. The resource name of the Execution to retrieve. Format: `p
      rojects/{project}/locations/{location}/metadataStores/{metadatastore}/ex
      ecutions/{execution}`
  """
    name = _messages.StringField(1, required=True)