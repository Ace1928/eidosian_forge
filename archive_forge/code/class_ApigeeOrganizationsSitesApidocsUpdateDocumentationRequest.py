from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsUpdateDocumentationRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsUpdateDocumentationRequest object.

  Fields:
    googleCloudApigeeV1ApiDocDocumentation: A
      GoogleCloudApigeeV1ApiDocDocumentation resource to be passed as the
      request body.
    name: Required. Resource name of the catalog item documentation. Use the
      following structure in your request:
      `organizations/{org}/sites/{site}/apidocs/{apidoc}/documentation`
  """
    googleCloudApigeeV1ApiDocDocumentation = _messages.MessageField('GoogleCloudApigeeV1ApiDocDocumentation', 1)
    name = _messages.StringField(2, required=True)