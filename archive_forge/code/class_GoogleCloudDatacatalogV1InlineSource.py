from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1InlineSource(_messages.Message):
    """Inline source containing taxonomies to import.

  Fields:
    taxonomies: Required. Taxonomies to import.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1SerializedTaxonomy', 1, repeated=True)