from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesPatchRequest object.

  Fields:
    googleCloudDatacatalogV1Entry: A GoogleCloudDatacatalogV1Entry resource to
      be passed as the request body.
    name: Output only. Identifier. The resource name of an entry in URL
      format. Note: The entry itself and its child resources might not be
      stored in the location specified in its name.
    updateMask: Names of fields whose values to overwrite on an entry. If this
      parameter is absent or empty, all modifiable fields are overwritten. If
      such fields are non-required and omitted in the request body, their
      values are emptied. You can modify only the fields listed below. For
      entries with type `DATA_STREAM`: * `schema` For entries with type
      `FILESET`: * `schema` * `display_name` * `description` *
      `gcs_fileset_spec` * `gcs_fileset_spec.file_patterns` For entries with
      `user_specified_type`: * `schema` * `display_name` * `description` *
      `user_specified_type` * `user_specified_system` * `linked_resource` *
      `source_system_timestamps`
  """
    googleCloudDatacatalogV1Entry = _messages.MessageField('GoogleCloudDatacatalogV1Entry', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)