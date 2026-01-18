from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresMetadataSchemasListRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresMetadataSchemasListRequest
  object.

  Fields:
    filter: A query to filter available MetadataSchemas for matching results.
    pageSize: The maximum number of MetadataSchemas to return. The service may
      return fewer. Must be in range 1-1000, inclusive. Defaults to 100.
    pageToken: A page token, received from a previous
      MetadataService.ListMetadataSchemas call. Provide this to retrieve the
      next page. When paginating, all other provided parameters must match the
      call that provided the page token. (Otherwise the request will fail with
      INVALID_ARGUMENT error.)
    parent: Required. The MetadataStore whose MetadataSchemas should be
      listed. Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)