from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresDeleteRequest object.

  Fields:
    force: Deprecated: Field is no longer supported.
    name: Required. The resource name of the MetadataStore to delete. Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)