from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3SerializedTaxonomy(_messages.Message):
    """Message capturing a taxonomy and its category hierachy as a nested
  proto. Used for taxonomy import/export and mutation.

  Fields:
    categories: Top level categories associated with the taxonomy if any.
    description: Description of the taxonomy. The length of the description is
      limited to 2000 bytes when encoded in UTF-8.
    displayName: Required. Display name of the taxonomy. Max 200 bytes when
      encoded in UTF-8.
  """
    categories = _messages.MessageField('GoogleCloudDatacatalogV1alpha3SerializedCategory', 1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)