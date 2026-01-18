from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1EntryGroup(_messages.Message):
    """EntryGroup Metadata. An EntryGroup resource represents a logical
  grouping of zero or more Data Catalog Entry resources.

  Fields:
    dataCatalogTimestamps: Output only. Timestamps about this EntryGroup.
      Default value is empty timestamps.
    description: Entry group description, which can consist of several
      sentences or paragraphs that describe entry group contents. Default
      value is an empty string.
    displayName: A short name to identify the entry group, for example,
      "analytics data - jan 2011". Default value is an empty string.
    name: Identifier. The resource name of the entry group in URL format.
      Example: *
      projects/{project_id}/locations/{location}/entryGroups/{entry_group_id}
      Note that this EntryGroup and its child resources may not actually be
      stored in the location in this name.
  """
    dataCatalogTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1beta1SystemTimestamps', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)