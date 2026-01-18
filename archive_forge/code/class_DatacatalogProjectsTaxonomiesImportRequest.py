from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesImportRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesImportRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3ImportTaxonomiesRequest: A
      GoogleCloudDatacatalogV1alpha3ImportTaxonomiesRequest resource to be
      passed as the request body.
    parent: Required. Resource name of project that the newly created
      taxonomies will belong to.
  """
    googleCloudDatacatalogV1alpha3ImportTaxonomiesRequest = _messages.MessageField('GoogleCloudDatacatalogV1alpha3ImportTaxonomiesRequest', 1)
    parent = _messages.StringField(2, required=True)