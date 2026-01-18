from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsCreateRequest object.

  Fields:
    googleCloudApigeeV1ApiProduct: A GoogleCloudApigeeV1ApiProduct resource to
      be passed as the request body.
    parent: Required. Name of the organization in which the API product will
      be created. Use the following structure in your request:
      `organizations/{org}`
  """
    googleCloudApigeeV1ApiProduct = _messages.MessageField('GoogleCloudApigeeV1ApiProduct', 1)
    parent = _messages.StringField(2, required=True)