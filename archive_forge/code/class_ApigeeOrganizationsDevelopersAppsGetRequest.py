from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsGetRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsGetRequest object.

  Fields:
    entity: **Note**: Must be used in conjunction with the `query` parameter.
      Set to `apiresources` to return the number of API resources that have
      been approved for access by a developer app in the specified Apigee
      organization.
    name: Required. Name of the developer app. Use the following structure in
      your request:
      `organizations/{org}/developers/{developer_email}/apps/{app}`
    query: **Note**: Must be used in conjunction with the `entity` parameter.
      Set to `count` to return the number of API resources that have been
      approved for access by a developer app in the specified Apigee
      organization.
  """
    entity = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    query = _messages.StringField(3)