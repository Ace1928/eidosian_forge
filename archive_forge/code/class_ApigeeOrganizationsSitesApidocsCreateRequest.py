from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsCreateRequest object.

  Fields:
    googleCloudApigeeV1ApiDoc: A GoogleCloudApigeeV1ApiDoc resource to be
      passed as the request body.
    parent: Required. Name of the portal. Use the following structure in your
      request: `organizations/{org}/sites/{site}`
  """
    googleCloudApigeeV1ApiDoc = _messages.MessageField('GoogleCloudApigeeV1ApiDoc', 1)
    parent = _messages.StringField(2, required=True)