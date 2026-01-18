from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsRateplansCreateRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsRateplansCreateRequest object.

  Fields:
    googleCloudApigeeV1RatePlan: A GoogleCloudApigeeV1RatePlan resource to be
      passed as the request body.
    parent: Required. Name of the API product that is associated with the rate
      plan. Use the following structure in your request:
      `organizations/{org}/apiproducts/{apiproduct}`
  """
    googleCloudApigeeV1RatePlan = _messages.MessageField('GoogleCloudApigeeV1RatePlan', 1)
    parent = _messages.StringField(2, required=True)