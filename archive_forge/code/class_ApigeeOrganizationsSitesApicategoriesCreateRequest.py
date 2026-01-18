from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApicategoriesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApicategoriesCreateRequest object.

  Fields:
    googleCloudApigeeV1ApiCategory: A GoogleCloudApigeeV1ApiCategory resource
      to be passed as the request body.
    parent: Required. Name of the portal. Use the following structure in your
      request: `organizations/{org}/sites/{site}`
  """
    googleCloudApigeeV1ApiCategory = _messages.MessageField('GoogleCloudApigeeV1ApiCategory', 1)
    parent = _messages.StringField(2, required=True)