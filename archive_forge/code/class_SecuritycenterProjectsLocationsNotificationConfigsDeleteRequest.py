from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsLocationsNotificationConfigsDeleteRequest(_messages.Message):
    """A SecuritycenterProjectsLocationsNotificationConfigsDeleteRequest
  object.

  Fields:
    name: Required. Name of the notification config to delete. The following
      list shows some examples of the format: + `organizations/[organization_i
      d]/locations/[location_id]/notificationConfigs/[config_id]` + `folders/[
      folder_id]/locations/[location_id]notificationConfigs/[config_id]` + `pr
      ojects/[project_id]/locations/[location_id]notificationConfigs/[config_i
      d]`
  """
    name = _messages.StringField(1, required=True)