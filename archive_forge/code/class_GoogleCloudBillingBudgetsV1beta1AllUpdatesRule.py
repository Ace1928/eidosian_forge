from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBillingBudgetsV1beta1AllUpdatesRule(_messages.Message):
    """AllUpdatesRule defines notifications that are sent based on budget spend
  and thresholds.

  Fields:
    disableDefaultIamRecipients: Optional. When set to true, disables default
      notifications sent when a threshold is exceeded. Default notifications
      are sent to those with Billing Account Administrator and Billing Account
      User IAM roles for the target account.
    enableProjectLevelRecipients: Optional. When set to true, and when the
      budget has a single project configured, notifications will be sent to
      project level recipients of that project. This field will be ignored if
      the budget has multiple or no project configured. Currently, project
      level recipients are the users with `Owner` role on a cloud project.
    monitoringNotificationChannels: Optional. Targets to send notifications to
      when a threshold is exceeded. This is in addition to default recipients
      who have billing account IAM roles. The value is the full REST resource
      name of a monitoring notification channel with the form
      `projects/{project_id}/notificationChannels/{channel_id}`. A maximum of
      5 channels are allowed. See https://cloud.google.com/billing/docs/how-
      to/budgets-notification-recipients for more details.
    pubsubTopic: Optional. The name of the Pub/Sub topic where budget related
      messages will be published, in the form
      `projects/{project_id}/topics/{topic_id}`. Updates are sent at regular
      intervals to the topic. The topic needs to be created before the budget
      is created; see https://cloud.google.com/billing/docs/how-to/budgets-
      programmatic-notifications for more details. Caller is expected to have
      `pubsub.topics.setIamPolicy` permission on the topic when it's set for a
      budget, otherwise, the API call will fail with PERMISSION_DENIED. See
      https://cloud.google.com/billing/docs/how-to/budgets-programmatic-
      notifications#permissions_required_for_this_task for more details on
      Pub/Sub roles and permissions.
    schemaVersion: Optional. Required when AllUpdatesRule.pubsub_topic is set.
      The schema version of the notification sent to
      AllUpdatesRule.pubsub_topic. Only "1.0" is accepted. It represents the
      JSON schema as defined in https://cloud.google.com/billing/docs/how-
      to/budgets-programmatic-notifications#notification_format.
  """
    disableDefaultIamRecipients = _messages.BooleanField(1)
    enableProjectLevelRecipients = _messages.BooleanField(2)
    monitoringNotificationChannels = _messages.StringField(3, repeated=True)
    pubsubTopic = _messages.StringField(4)
    schemaVersion = _messages.StringField(5)