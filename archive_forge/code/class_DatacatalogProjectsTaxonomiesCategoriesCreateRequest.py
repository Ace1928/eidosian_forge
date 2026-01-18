from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesCategoriesCreateRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesCategoriesCreateRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3Category: A
      GoogleCloudDatacatalogV1alpha3Category resource to be passed as the
      request body.
    parent: Required. Resource name of the taxonomy that the newly created
      category belongs to.
  """
    googleCloudDatacatalogV1alpha3Category = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Category', 1)
    parent = _messages.StringField(2, required=True)