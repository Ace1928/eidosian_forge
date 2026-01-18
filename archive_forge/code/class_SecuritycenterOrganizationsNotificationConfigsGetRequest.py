from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsNotificationConfigsGetRequest(_messages.Message):
    """A SecuritycenterOrganizationsNotificationConfigsGetRequest object.

  Fields:
    name: Required. Name of the notification config to get. Its format is
      "organizations/[organization_id]/notificationConfigs/[config_id]",
      "folders/[folder_id]/notificationConfigs/[config_id]", or
      "projects/[project_id]/notificationConfigs/[config_id]".
  """
    name = _messages.StringField(1, required=True)