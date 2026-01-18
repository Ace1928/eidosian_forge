from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VisionProjectsLocationsProductSetsRemoveProductRequest(_messages.Message):
    """A VisionProjectsLocationsProductSetsRemoveProductRequest object.

  Fields:
    name: Required. The resource name for the ProductSet to modify. Format is:
      `projects/PROJECT_ID/locations/LOC_ID/productSets/PRODUCT_SET_ID`
    removeProductFromProductSetRequest: A RemoveProductFromProductSetRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    removeProductFromProductSetRequest = _messages.MessageField('RemoveProductFromProductSetRequest', 2)