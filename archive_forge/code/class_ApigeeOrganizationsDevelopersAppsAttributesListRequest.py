from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsAttributesListRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsAttributesListRequest object.

  Fields:
    parent: Required. Name of the developer app. Use the following structure
      in your request:
      `organizations/{org}/developers/{developer_email}/apps/{app}`
  """
    parent = _messages.StringField(1, required=True)