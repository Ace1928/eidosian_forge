from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1alpha3ImportTaxonomiesResponse(_messages.Message):
    """Response message for "CategoryManagerSerialization.ImportTaxonomies".

  Fields:
    taxonomies: Required. Taxonomies that were imported.
  """
    taxonomies = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Taxonomy', 1, repeated=True)