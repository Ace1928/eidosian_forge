from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterProjectsLocationsNotificationConfigsCreateRequest(_messages.Message):
    """A SecuritycenterProjectsLocationsNotificationConfigsCreateRequest
  object.

  Fields:
    configId: Required. Unique identifier provided by the client within the
      parent scope. It must be between 1 and 128 characters and contain
      alphanumeric characters, underscores, or hyphens only.
    notificationConfig: A NotificationConfig resource to be passed as the
      request body.
    parent: Required. Resource name of the new notification config's parent.
      Its format is "organizations/[organization_id]/locations/[location_id]",
      "folders/[folder_id]/locations/[location_id]", or
      "projects/[project_id]/locations/[location_id]".
  """
    configId = _messages.StringField(1)
    notificationConfig = _messages.MessageField('NotificationConfig', 2)
    parent = _messages.StringField(3, required=True)