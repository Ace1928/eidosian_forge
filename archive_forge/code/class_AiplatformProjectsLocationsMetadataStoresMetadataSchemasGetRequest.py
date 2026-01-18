from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresMetadataSchemasGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresMetadataSchemasGetRequest
  object.

  Fields:
    name: Required. The resource name of the MetadataSchema to retrieve.
      Format: `projects/{project}/locations/{location}/metadataStores/{metadat
      astore}/metadataSchemas/{metadataschema}`
  """
    name = _messages.StringField(1, required=True)