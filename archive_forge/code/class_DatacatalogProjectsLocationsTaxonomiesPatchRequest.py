from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTaxonomiesPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTaxonomiesPatchRequest object.

  Fields:
    googleCloudDatacatalogV1Taxonomy: A GoogleCloudDatacatalogV1Taxonomy
      resource to be passed as the request body.
    name: Identifier. Resource name of this taxonomy in URL format. Note:
      Policy tag manager generates unique taxonomy IDs.
    updateMask: Specifies fields to update. If not set, defaults to all fields
      you can update. For more information, see [FieldMask]
      (https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask).
  """
    googleCloudDatacatalogV1Taxonomy = _messages.MessageField('GoogleCloudDatacatalogV1Taxonomy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)