from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsKeysApiproductsUpdateAppGroupAppKeyApiProductRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsKeysApiproductsUpdateAppGroupAppKeyApi
  ProductRequest object.

  Fields:
    action: Approve or revoke the consumer key by setting this value to
      `approve` or `revoke` respectively. The `Content-Type` header, if set,
      must be set to `application/octet-stream`, with empty body.
    name: Required. Name of the API product in the developer app key in the
      following format: `organizations/{org}/appgroups/{app_group_name}/apps/{
      app}/keys/{key}/apiproducts/{apiproduct}`
  """
    action = _messages.StringField(1)
    name = _messages.StringField(2, required=True)