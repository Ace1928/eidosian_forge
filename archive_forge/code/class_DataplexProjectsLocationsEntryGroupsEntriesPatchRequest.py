from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsEntriesPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsEntriesPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true and the entry does not exist, it
      will be created.
    aspectKeys: Optional. The map keys of the Aspects which should be
      modified. Supports the following syntaxes: * - matches aspect on given
      type and empty path * @path - matches aspect on given type and specified
      path * * - matches aspects on given type for all paths * *@path -
      matches aspects of all types on the given pathExisting aspects matching
      the syntax will not be removed unless delete_missing_aspects is set to
      true.If this field is left empty, it will be treated as specifying
      exactly those Aspects present in the request.
    deleteMissingAspects: Optional. If set to true and the aspect_keys specify
      aspect ranges, any existing aspects from that range not provided in the
      request will be deleted.
    googleCloudDataplexV1Entry: A GoogleCloudDataplexV1Entry resource to be
      passed as the request body.
    name: Identifier. The relative resource name of the Entry, of the form: pr
      ojects/{project}/locations/{location}/entryGroups/{entry_group}/entries/
      {entry}.
    updateMask: Optional. Mask of fields to update. To update Aspects, the
      update_mask must contain the value "aspects".If the update_mask is
      empty, all modifiable fields present in the request will be updated.
  """
    allowMissing = _messages.BooleanField(1)
    aspectKeys = _messages.StringField(2, repeated=True)
    deleteMissingAspects = _messages.BooleanField(3)
    googleCloudDataplexV1Entry = _messages.MessageField('GoogleCloudDataplexV1Entry', 4)
    name = _messages.StringField(5, required=True)
    updateMask = _messages.StringField(6)