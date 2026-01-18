from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresExecutionsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresExecutionsDeleteRequest
  object.

  Fields:
    etag: Optional. The etag of the Execution to delete. If this is provided,
      it must match the server's etag. Otherwise, the request will fail with a
      FAILED_PRECONDITION.
    name: Required. The resource name of the Execution to delete. Format: `pro
      jects/{project}/locations/{location}/metadataStores/{metadatastore}/exec
      utions/{execution}`
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)