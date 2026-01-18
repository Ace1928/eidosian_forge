from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesCategoriesPatchRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesCategoriesPatchRequest object.

  Fields:
    googleCloudDatacatalogV1alpha3Category: A
      GoogleCloudDatacatalogV1alpha3Category resource to be passed as the
      request body.
    name: Required. Resource name of the category to be updated.
    updateMask: The update mask applies to the resource. Only display_name,
      description and parent_category_id can be updated and thus can be listed
      in the mask. If update_mask is not provided, all allowed fields (i.e.
      display_name, description and parent_id) will be updated. For more
      information including the `FieldMask` definition, see
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    googleCloudDatacatalogV1alpha3Category = _messages.MessageField('GoogleCloudDatacatalogV1alpha3Category', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)