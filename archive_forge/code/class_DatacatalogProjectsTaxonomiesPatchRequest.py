from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesPatchRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesPatchRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3Taxonomy: A
      GoogleCloudDatacatalogV1alpha3Taxonomy resource to be passed as the
      request body.
    name: Required. Resource name of the taxonomy to be updated.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    googleCloudDatacatalogV1alpha3Taxonomy = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Taxonomy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)