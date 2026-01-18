from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAppsKeysUpdateDeveloperAppKeyRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAppsKeysUpdateDeveloperAppKeyRequest
  object.

  Fields:
    action: Approve or revoke the consumer key by setting this value to
      `approve` or `revoke`, respectively. The `Content-Type` header must be
      set to `application/octet-stream`.
    googleCloudApigeeV1DeveloperAppKey: A GoogleCloudApigeeV1DeveloperAppKey
      resource to be passed as the request body.
    name: Name of the developer app key. Use the following structure in your
      request:
      `organizations/{org}/developers/{developer_email}/apps/{app}/keys/{key}`
  """
    action = _messages.StringField(1)
    googleCloudApigeeV1DeveloperAppKey = _messages.MessageField('GoogleCloudApigeeV1DeveloperAppKey', 2)
    name = _messages.StringField(3, required=True)