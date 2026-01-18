from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsUpdateRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsUpdateRequest object.

  Fields:
    action: Activate or de-activate the AppGroup by setting the action as
      `active` or `inactive`. The `Content-Type` header must be set to
      `application/octet-stream`, with empty body.
    googleCloudApigeeV1AppGroup: A GoogleCloudApigeeV1AppGroup resource to be
      passed as the request body.
    name: Required. Name of the AppGroup. Use the following structure in your
      request: `organizations/{org}/appgroups/{app_group_name}`
  """
    action = _messages.StringField(1)
    googleCloudApigeeV1AppGroup = _messages.MessageField('GoogleCloudApigeeV1AppGroup', 2)
    name = _messages.StringField(3, required=True)