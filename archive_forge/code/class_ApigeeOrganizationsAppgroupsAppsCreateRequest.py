from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsAppgroupsAppsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsAppgroupsAppsCreateRequest object.

  Fields:
    googleCloudApigeeV1AppGroupApp: A GoogleCloudApigeeV1AppGroupApp resource
      to be passed as the request body.
    parent: Required. Name of the AppGroup. Use the following structure in
      your request: `organizations/{org}/appgroups/{app_group_name}`
  """
    googleCloudApigeeV1AppGroupApp = _messages.MessageField('GoogleCloudApigeeV1AppGroupApp', 1)
    parent = _messages.StringField(2, required=True)