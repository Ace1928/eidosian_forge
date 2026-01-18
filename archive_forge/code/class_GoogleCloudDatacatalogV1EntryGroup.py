from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1EntryGroup(_messages.Message):
    """Entry group metadata. An `EntryGroup` resource represents a logical
  grouping of zero or more Data Catalog Entry resources.

  Fields:
    dataCatalogTimestamps: Output only. Timestamps of the entry group. Default
      value is empty.
    description: Entry group description. Can consist of several sentences or
      paragraphs that describe the entry group contents. Default value is an
      empty string.
    displayName: A short name to identify the entry group, for example,
      "analytics data - jan 2011". Default value is an empty string.
    name: Identifier. The resource name of the entry group in URL format.
      Note: The entry group itself and its child resources might not be stored
      in the location specified in its name.
  """
    dataCatalogTimestamps = _messages.MessageField('GoogleCloudDatacatalogV1SystemTimestamps', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)