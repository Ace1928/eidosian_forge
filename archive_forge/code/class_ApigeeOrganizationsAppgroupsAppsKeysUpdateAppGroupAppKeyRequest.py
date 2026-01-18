from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsKeysUpdateAppGroupAppKeyRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsKeysUpdateAppGroupAppKeyRequest
  object.

  Fields:
    googleCloudApigeeV1UpdateAppGroupAppKeyRequest: A
      GoogleCloudApigeeV1UpdateAppGroupAppKeyRequest resource to be passed as
      the request body.
    name: Required. Name of the AppGroup app key. Use the following structure
      in your request:
      `organizations/{org}/appgroups/{app_group_name}/apps/{app}/keys/{key}`
  """
    googleCloudApigeeV1UpdateAppGroupAppKeyRequest = _messages.MessageField('GoogleCloudApigeeV1UpdateAppGroupAppKeyRequest', 1)
    name = _messages.StringField(2, required=True)