from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesTagsPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesTagsPatchRequest object.

  Fields:
    googleCloudDatacatalogV1Tag: A GoogleCloudDatacatalogV1Tag resource to be
      passed as the request body.
    name: Identifier. The resource name of the tag in URL format where tag ID
      is a system-generated identifier. Note: The tag itself might not be
      stored in the location specified in its name.
    updateMask: Names of fields whose values to overwrite on a tag. Currently,
      a tag has the only modifiable field with the name `fields`. In general,
      if this parameter is absent or empty, all modifiable fields are
      overwritten. If such fields are non-required and omitted in the request
      body, their values are emptied.
  """
    googleCloudDatacatalogV1Tag = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)