from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsGetDocumentationRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsGetDocumentationRequest object.

  Fields:
    name: Required. Resource name of the catalog item documentation. Use the
      following structure in your request:
      `organizations/{org}/sites/{site}/apidocs/{apidoc}/documentation`
  """
    name = _messages.StringField(1, required=True)