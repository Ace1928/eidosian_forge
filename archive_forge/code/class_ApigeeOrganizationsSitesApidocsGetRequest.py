from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSitesApidocsGetRequest(_messages.Message):
    """A ApigeeOrganizationsSitesApidocsGetRequest object.

  Fields:
    name: Required. Name of the catalog item. Use the following structure in
      your request: `organizations/{org}/sites/{site}/apidocs/{apidoc}`
  """
    name = _messages.StringField(1, required=True)