from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudEssentialcontactsV1alpha1Contact(_messages.Message):
    """A contact that will receive notifications from Google Cloud.

  Enums:
    NotificationCategorySubscriptionsValueListEntryValuesEnum:
    ValidationStateValueValuesEnum: The validity of the contact. A contact is
      considered valid if it is the correct recipient for notifications for a
      particular resource.
    VerificationStateValueValuesEnum: The verification state of this contact's
      email address.

  Fields:
    email: Required. The email address to send notifications to. The email
      address does not need to be a Google Account.
    languageTag: Required. The preferred language for notifications, as a ISO
      639-1 language code. See [Supported
      languages](https://cloud.google.com/resource-manager/docs/managing-
      notification-contacts#supported-languages) for a list of supported
      languages.
    name: Output only. The identifier for the contact. Format:
      {resource_type}/{resource_id}/contacts/{contact_id}
    notificationCategorySubscriptions: Required. The categories of
      notifications that the contact will receive communications for.
    validateTime: The last time the validation_state was updated, either
      manually or automatically. A contact is considered stale if its
      validation state was updated more than 1 year ago.
    validationState: The validity of the contact. A contact is considered
      valid if it is the correct recipient for notifications for a particular
      resource.
    verificationExpireTime: Time when the current verification token will
      expire. After this a new token will need to be generated for the user to
      verify the contact.
    verificationState: The verification state of this contact's email address.
  """

    class NotificationCategorySubscriptionsValueListEntryValuesEnum(_messages.Enum):
        """NotificationCategorySubscriptionsValueListEntryValuesEnum enum type.

    Values:
      NOTIFICATION_CATEGORY_UNSPECIFIED: Notification category is unrecognized
        or unspecified.
      OTHER: Deprecated. Please use a more specific category.
      ALL: All notifications related to the resource, including notifications
        pertaining to categories added in the future.
      SUSPENSION: Notifications related to imminent account suspension.
      PRIVACY: Deprecated. Please use security instead.
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
        OTHER = 1
        ALL = 2
        SUSPENSION = 3
        PRIVACY = 4
        SECURITY = 5
        TECHNICAL = 6
        BILLING = 7
        LEGAL = 8
        PRODUCT_UPDATES = 9
        TECHNICAL_INCIDENTS = 10

    class ValidationStateValueValuesEnum(_messages.Enum):
        """The validity of the contact. A contact is considered valid if it is
    the correct recipient for notifications for a particular resource.

    Values:
      VALIDATION_STATE_UNSPECIFIED: The validation state is unknown or
        unspecified.
      VALID: The contact is marked as valid. This is usually done manually by
        the contact admin. All new contacts begin in the valid state.
      INVALID: The contact is considered invalid. This may become the state if
        the contact's email is found to be unreachable.
    """
        VALIDATION_STATE_UNSPECIFIED = 0
        VALID = 1
        INVALID = 2

    class VerificationStateValueValuesEnum(_messages.Enum):
        """The verification state of this contact's email address.

    Values:
      VERIFICATION_STATE_UNSPECIFIED: VerificationState is unrecognized or
        unspecified.
      PENDING: Verification was sent but has not been accepted yet. A contact
        will remain in this state even if verification time limit elapses. At
        that point the contact cannot be verified, and ResendVerification must
        be called to reset the timeout.
      VERIFIED: Email has been verified.
      FAILED: Error with verification - email could not be delivered.
    """
        VERIFICATION_STATE_UNSPECIFIED = 0
        PENDING = 1
        VERIFIED = 2
        FAILED = 3
    email = _messages.StringField(1)
    languageTag = _messages.StringField(2)
    name = _messages.StringField(3)
    notificationCategorySubscriptions = _messages.EnumField('NotificationCategorySubscriptionsValueListEntryValuesEnum', 4, repeated=True)
    validateTime = _messages.StringField(5)
    validationState = _messages.EnumField('ValidationStateValueValuesEnum', 6)
    verificationExpireTime = _messages.StringField(7)
    verificationState = _messages.EnumField('VerificationStateValueValuesEnum', 8)