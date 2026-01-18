from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsRateplansDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsRateplansDeleteRequest object.

  Fields:
    name: Required. ID of the rate plan. Use the following structure in your
      request:
      `organizations/{org}/apiproducts/{apiproduct}/rateplans/{rateplan}`
  """
    name = _messages.StringField(1, required=True)