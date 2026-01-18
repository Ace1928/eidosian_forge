from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsLocationsNotificationConfigsPatchRequest(_messages.Message):
    """A SecuritycenterOrganizationsLocationsNotificationConfigsPatchRequest
  object.

  Fields:
    name: The relative resource name of this notification config. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me The following list shows some examples: + `organizations/{organizatio
      n_id}/locations/{location_id}/notificationConfigs/notify_public_bucket`
      + `folders/{folder_id}/locations/{location_id}/notificationConfigs/notif
      y_public_bucket` + `projects/{project_id}/locations/{location_id}/notifi
      cationConfigs/notify_public_bucket`
    notificationConfig: A NotificationConfig resource to be passed as the
      request body.
    updateMask: The FieldMask to use when updating the notification config. If
      empty all mutable fields will be updated.
  """
    name = _messages.StringField(1, required=True)
    notificationConfig = _messages.MessageField('NotificationConfig', 2)
    updateMask = _messages.StringField(3)