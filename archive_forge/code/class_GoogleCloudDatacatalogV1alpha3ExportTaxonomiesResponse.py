from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ExportTaxonomiesResponse(_messages.Message):
    """Response message for "CategoryManagerSerialization.ExportTaxonomies".

  Fields:
    taxonomies: Required. List of taxonomies and categories in a tree
      structure.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1alpha3SerializedTaxonomy', 1, repeated=True)