from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsUpdateRequest object.

  Fields:
    googleCloudApigeeV1ApiDoc: A GoogleCloudApigeeV1ApiDoc resource to be
      passed as the request body.
    name: Required. Name of the catalog item. Use the following structure in
      your request: `organizations/{org}/sites/{site}/apidocs/{apidoc}`
  """
    googleCloudApigeeV1ApiDoc = _messages.MessageField('GoogleCloudApigeeV1ApiDoc', 1)
    name = _messages.StringField(2, required=True)