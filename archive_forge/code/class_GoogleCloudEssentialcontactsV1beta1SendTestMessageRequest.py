from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudEssentialcontactsV1beta1SendTestMessageRequest(_messages.Message):
    """Request message for the SendTestMessage method.

  Enums:
    NotificationCategoryValueValuesEnum: Required. The notification category
      to send the test message for. All contacts must be subscribed to this
      category.

  Fields:
    contacts: Required. The list of names of the contacts to send a test
      message to. Format:
      organizations/{organization_id}/contacts/{contact_id},
      folders/{folder_id}/contacts/{contact_id} or
      projects/{project_id}/contacts/{contact_id}
    notificationCategory: Required. The notification category to send the test
      message for. All contacts must be subscribed to this category.
  """

    class NotificationCategoryValueValuesEnum(_messages.Enum):
        """Required. The notification category to send the test message for. All
    contacts must be subscribed to this category.

    Values:
      NOTIFICATION_CATEGORY_UNSPECIFIED: Notification category is unrecognized
        or unspecified.
      ALL: All notifications related to the resource, including notifications
        pertaining to categories added in the future.
      SUSPENSION: Notifications related to imminent account suspension.
      SECURITY: Notifications related to security/privacy incidents,
        notifications, and vulnerabilities.
      TECHNICAL: Notifications related to technical events and issues such as
        outages, errors, or bugs.
      BILLING: Notifications related to billing and payments notifications,
        price updates, errors, or credits.
      LEGAL: Notifications related to enforcement actions, regulatory
        compliance, or government notices.
      PRODUCT_UPDATES: Notifications related to new versions, product terms
        updates, or deprecations.
      TECHNICAL_INCIDENTS: Child category of TECHNICAL. If assigned, technical
        incident notifications will go to these contacts instead of TECHNICAL.
    """
        NOTIFICATION_CATEGORY_UNSPECIFIED = 0
        ALL = 1
        SUSPENSION = 2
        SECURITY = 3
        TECHNICAL = 4
        BILLING = 5
        LEGAL = 6
        PRODUCT_UPDATES = 7
        TECHNICAL_INCIDENTS = 8
    contacts = _messages.StringField(1, repeated=True)
    notificationCategory = _messages.EnumField('NotificationCategoryValueValuesEnum', 2)