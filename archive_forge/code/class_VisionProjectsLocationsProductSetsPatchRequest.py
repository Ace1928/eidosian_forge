from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsPatchRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsPatchRequest object.

  Fields:
    name: The resource name of the ProductSet. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/productSets/PRODUCT_SET_ID`. This
      field is ignored when creating a ProductSet.
    productSet: A ProductSet resource to be passed as the request body.
    updateMask: The FieldMask that specifies which fields to update. If
      update_mask isn't specified, all mutable fields are to be updated. Valid
      mask path is `display_name`.
  """
    name = _messages.StringField(1, required=True)
    productSet = _messages.MessageField('ProductSet', 2)
    updateMask = _messages.StringField(3)