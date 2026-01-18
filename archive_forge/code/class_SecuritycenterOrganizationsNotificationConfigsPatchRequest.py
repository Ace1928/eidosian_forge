from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterOrganizationsNotificationConfigsPatchRequest(_messages.Message):
    """A SecuritycenterOrganizationsNotificationConfigsPatchRequest object.

  Fields:
    name: The relative resource name of this notification config. See:
      https://cloud.google.com/apis/design/resource_names#relative_resource_na
      me Example: "organizations/{organization_id}/notificationConfigs/notify_
      public_bucket",
      "folders/{folder_id}/notificationConfigs/notify_public_bucket", or
      "projects/{project_id}/notificationConfigs/notify_public_bucket".
    notificationConfig: A NotificationConfig resource to be passed as the
      request body.
    updateMask: The FieldMask to use when updating the notification config. If
      empty all mutable fields will be updated.
  """
    name = _messages.StringField(1, required=True)
    notificationConfig = _messages.MessageField('NotificationConfig', 2)
    updateMask = _messages.StringField(3)