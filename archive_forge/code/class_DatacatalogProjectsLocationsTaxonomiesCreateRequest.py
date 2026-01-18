from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesCreateRequest object.

  Fields:
    googleCloudDatacatalogV1Taxonomy: A GoogleCloudDatacatalogV1Taxonomy
      resource to be passed as the request body.
    parent: Required. Resource name of the project that the taxonomy will
      belong to.
  """
    googleCloudDatacatalogV1Taxonomy = _messages.MessageField('GoogleCloudDatacatalogV1Taxonomy', 1)
    parent = _messages.StringField(2, required=True)