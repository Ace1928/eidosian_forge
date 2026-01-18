from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsLocationsNotificationConfigsGetRequest(_messages.Message):
    """A SecuritycenterOrganizationsLocationsNotificationConfigsGetRequest
  object.

  Fields:
    name: Required. Name of the notification config to get. The following list
      shows some examples of the format: + `organizations/[organization_id]/lo
      cations/[location_id]/notificationConfigs/[config_id]` + `folders/[folde
      r_id]/locations/[location_id]/notificationConfigs/[config_id]` + `projec
      ts/[project_id]/locations/[location_id]/notificationConfigs/[config_id]`
  """
    name = _messages.StringField(1, required=True)