from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Registration(_messages.Message):
    """The `Registration` resource facilitates managing and configuring domain
  name registrations. There are several ways to create a new `Registration`
  resource: To create a new `Registration` resource, find a suitable domain
  name by calling the `SearchDomains` method with a query to see available
  domain name options. After choosing a name, call
  `RetrieveRegisterParameters` to ensure availability and obtain information
  like pricing, which is needed to build a call to `RegisterDomain`. Another
  way to create a new `Registration` is to transfer an existing domain from
  another registrar (Deprecated: For more information, see [Cloud Domains
  feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations)). First, go to the current registrar to unlock the domain for
  transfer and retrieve the domain's transfer authorization code. Then call
  `RetrieveTransferParameters` to confirm that the domain is unlocked and to
  get values needed to build a call to `TransferDomain`. Finally, you can
  create a new `Registration` by importing an existing domain managed with
  [Google Domains](https://domains.google/) (Deprecated: For more information,
  see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations)). First, call `RetrieveImportableDomains` to list domains to
  which the calling user has sufficient access. Then call `ImportDomain` on
  any domain names you want to use with Cloud Domains.

  Enums:
    IssuesValueListEntryValuesEnum:
    ProviderValueValuesEnum: Output only. Current domain management provider.
    RegisterFailureReasonValueValuesEnum: Output only. The reason the domain
      registration failed. Only set for domains in REGISTRATION_FAILED state.
    StateValueValuesEnum: Output only. The state of the `Registration`
    SupportedPrivacyValueListEntryValuesEnum:
    TransferFailureReasonValueValuesEnum: Output only. Deprecated: For more
      information, see [Cloud Domains feature
      deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
      deprecations). The reason the domain transfer failed. Only set for
      domains in TRANSFER_FAILED state.

  Messages:
    LabelsValue: Set of labels associated with the `Registration`.

  Fields:
    contactSettings: Required. Settings for contact information linked to the
      `Registration`. You cannot update these with the `UpdateRegistration`
      method. To update these settings, use the `ConfigureContactSettings`
      method.
    createTime: Output only. The creation timestamp of the `Registration`
      resource.
    dnsSettings: Settings controlling the DNS configuration of the
      `Registration`. You cannot update these with the `UpdateRegistration`
      method. To update these settings, use the `ConfigureDnsSettings` method.
    domainName: Required. Immutable. The domain name. Unicode domain names
      must be expressed in Punycode format.
    expireTime: Output only. The expiration timestamp of the `Registration`.
    issues: Output only. The set of issues with the `Registration` that
      require attention.
    labels: Set of labels associated with the `Registration`.
    managementSettings: Settings for management of the `Registration`,
      including renewal, billing, and transfer. You cannot update these with
      the `UpdateRegistration` method. To update these settings, use the
      `ConfigureManagementSettings` method.
    name: Output only. Name of the `Registration` resource, in the format
      `projects/*/locations/*/registrations/`.
    pendingContactSettings: Output only. Pending contact settings for the
      `Registration`. Updates to the `contact_settings` field that change its
      `registrant_contact` or `privacy` fields require email confirmation by
      the `registrant_contact` before taking effect. This field is set only if
      there are pending updates to the `contact_settings` that have not been
      confirmed. To confirm the changes, the `registrant_contact` must follow
      the instructions in the email they receive.
    provider: Output only. Current domain management provider.
    registerFailureReason: Output only. The reason the domain registration
      failed. Only set for domains in REGISTRATION_FAILED state.
    state: Output only. The state of the `Registration`
    supportedPrivacy: Output only. Set of options for the
      `contact_settings.privacy` field that this `Registration` supports.
    transferFailureReason: Output only. Deprecated: For more information, see
      [Cloud Domains feature
      deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
      deprecations). The reason the domain transfer failed. Only set for
      domains in TRANSFER_FAILED state.
  """

    class IssuesValueListEntryValuesEnum(_messages.Enum):
        """IssuesValueListEntryValuesEnum enum type.

    Values:
      ISSUE_UNSPECIFIED: The issue is undefined.
      CONTACT_SUPPORT: Contact the Cloud Support team to resolve a problem
        with this domain.
      UNVERIFIED_EMAIL: [ICANN](https://icann.org/) requires verification of
        the email address in the `Registration`'s
        `contact_settings.registrant_contact` field. To verify the email
        address, follow the instructions in the email the `registrant_contact`
        receives following registration. If you do not complete email
        verification within 15 days of registration, the domain is suspended.
        To resend the verification email, call ConfigureContactSettings and
        provide the current `registrant_contact.email`.
      PROBLEM_WITH_BILLING: The billing account is not in good standing. The
        domain is not automatically renewed at its expiration time unless you
        resolve problems with your billing account.
    """
        ISSUE_UNSPECIFIED = 0
        CONTACT_SUPPORT = 1
        UNVERIFIED_EMAIL = 2
        PROBLEM_WITH_BILLING = 3

    class ProviderValueValuesEnum(_messages.Enum):
        """Output only. Current domain management provider.

    Values:
      REGISTRAR_UNSPECIFIED: Registrar is not selected.
      GOOGLE_DOMAINS: Use Google Domains registrar.
      SQUARESPACE: Use Squarespace registrar
    """
        REGISTRAR_UNSPECIFIED = 0
        GOOGLE_DOMAINS = 1
        SQUARESPACE = 2

    class RegisterFailureReasonValueValuesEnum(_messages.Enum):
        """Output only. The reason the domain registration failed. Only set for
    domains in REGISTRATION_FAILED state.

    Values:
      REGISTER_FAILURE_REASON_UNSPECIFIED: Register failure unspecified.
      REGISTER_FAILURE_REASON_UNKNOWN: Registration failed for an unknown
        reason.
      DOMAIN_NOT_AVAILABLE: The domain is not available for registration.
      INVALID_CONTACTS: The provided contact information was rejected.
    """
        REGISTER_FAILURE_REASON_UNSPECIFIED = 0
        REGISTER_FAILURE_REASON_UNKNOWN = 1
        DOMAIN_NOT_AVAILABLE = 2
        INVALID_CONTACTS = 3

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the `Registration`

    Values:
      STATE_UNSPECIFIED: The state is undefined.
      REGISTRATION_PENDING: The domain is being registered.
      REGISTRATION_FAILED: The domain registration failed. You can delete
        resources in this state to allow registration to be retried.
      TRANSFER_PENDING: The domain is being transferred from another registrar
        to Cloud Domains.
      TRANSFER_FAILED: The attempt to transfer the domain from another
        registrar to Cloud Domains failed. You can delete resources in this
        state and retry the transfer.
      IMPORT_PENDING: The domain is being imported from Google Domains to
        Cloud Domains.
      ACTIVE: The domain is registered and operational. The domain renews
        automatically as long as it remains in this state and the
        `RenewalMethod` is set to `AUTOMATIC_RENEWAL`.
      SUSPENDED: The domain is suspended and inoperative. For more details,
        see the `issues` field.
      EXPORTED: The domain is no longer managed with Cloud Domains. It may
        have been transferred to another registrar or exported for management
        in [Google Domains](https://domains.google/). You can no longer update
        it with this API, and information shown about it may be stale. Domains
        in this state are not automatically renewed by Cloud Domains.
      EXPIRED: The domain is expired.
    """
        STATE_UNSPECIFIED = 0
        REGISTRATION_PENDING = 1
        REGISTRATION_FAILED = 2
        TRANSFER_PENDING = 3
        TRANSFER_FAILED = 4
        IMPORT_PENDING = 5
        ACTIVE = 6
        SUSPENDED = 7
        EXPORTED = 8
        EXPIRED = 9

    class SupportedPrivacyValueListEntryValuesEnum(_messages.Enum):
        """SupportedPrivacyValueListEntryValuesEnum enum type.

    Values:
      CONTACT_PRIVACY_UNSPECIFIED: The contact privacy settings are undefined.
      PUBLIC_CONTACT_DATA: All the data from `ContactSettings` is publicly
        available. When setting this option, you must also provide a
        `PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT` in the `contact_notices` field
        of the request.
      PRIVATE_CONTACT_DATA: Deprecated: For more information, see [Cloud
        Domains feature deprecation](https://cloud.google.com/domains/docs/dep
        recations/feature-deprecations). None of the data from
        `ContactSettings` is publicly available. Instead, proxy contact data
        is published for your domain. Email sent to the proxy email address is
        forwarded to the registrant's email address. Cloud Domains provides
        this privacy proxy service at no additional cost.
      REDACTED_CONTACT_DATA: The organization name (if provided) and limited
        non-identifying data from `ContactSettings` is available to the public
        (e.g. country and state). The remaining data is marked as `REDACTED
        FOR PRIVACY` in the WHOIS database. The actual information redacted
        depends on the domain. For details, see [the registration privacy
        article](https://support.google.com/domains/answer/3251242).
    """
        CONTACT_PRIVACY_UNSPECIFIED = 0
        PUBLIC_CONTACT_DATA = 1
        PRIVATE_CONTACT_DATA = 2
        REDACTED_CONTACT_DATA = 3

    class TransferFailureReasonValueValuesEnum(_messages.Enum):
        """Output only. Deprecated: For more information, see [Cloud Domains
    feature
    deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
    deprecations). The reason the domain transfer failed. Only set for domains
    in TRANSFER_FAILED state.

    Values:
      TRANSFER_FAILURE_REASON_UNSPECIFIED: Transfer failure unspecified.
      TRANSFER_FAILURE_REASON_UNKNOWN: Transfer failed for an unknown reason.
      EMAIL_CONFIRMATION_FAILURE: An email confirmation sent to the user was
        rejected or expired.
      DOMAIN_NOT_REGISTERED: The domain is available for registration.
      DOMAIN_HAS_TRANSFER_LOCK: The domain has a transfer lock with its
        current registrar which must be removed prior to transfer.
      INVALID_AUTHORIZATION_CODE: The authorization code entered is not valid.
      TRANSFER_CANCELLED: The transfer was cancelled by the domain owner,
        current registrar, or TLD registry.
      TRANSFER_REJECTED: The transfer was rejected by the current registrar.
        Contact the current registrar for more information.
      INVALID_REGISTRANT_EMAIL_ADDRESS: The registrant email address cannot be
        parsed from the domain's current public contact data.
      DOMAIN_NOT_ELIGIBLE_FOR_TRANSFER: The domain is not eligible for
        transfer due requirements imposed by the current registrar or TLD
        registry.
      TRANSFER_ALREADY_PENDING: Another transfer is already pending for this
        domain. The existing transfer attempt must expire or be cancelled in
        order to proceed.
    """
        TRANSFER_FAILURE_REASON_UNSPECIFIED = 0
        TRANSFER_FAILURE_REASON_UNKNOWN = 1
        EMAIL_CONFIRMATION_FAILURE = 2
        DOMAIN_NOT_REGISTERED = 3
        DOMAIN_HAS_TRANSFER_LOCK = 4
        INVALID_AUTHORIZATION_CODE = 5
        TRANSFER_CANCELLED = 6
        TRANSFER_REJECTED = 7
        INVALID_REGISTRANT_EMAIL_ADDRESS = 8
        DOMAIN_NOT_ELIGIBLE_FOR_TRANSFER = 9
        TRANSFER_ALREADY_PENDING = 10

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with the `Registration`.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    contactSettings = _messages.MessageField('ContactSettings', 1)
    createTime = _messages.StringField(2)
    dnsSettings = _messages.MessageField('DnsSettings', 3)
    domainName = _messages.StringField(4)
    expireTime = _messages.StringField(5)
    issues = _messages.EnumField('IssuesValueListEntryValuesEnum', 6, repeated=True)
    labels = _messages.MessageField('LabelsValue', 7)
    managementSettings = _messages.MessageField('ManagementSettings', 8)
    name = _messages.StringField(9)
    pendingContactSettings = _messages.MessageField('ContactSettings', 10)
    provider = _messages.EnumField('ProviderValueValuesEnum', 11)
    registerFailureReason = _messages.EnumField('RegisterFailureReasonValueValuesEnum', 12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    supportedPrivacy = _messages.EnumField('SupportedPrivacyValueListEntryValuesEnum', 14, repeated=True)
    transferFailureReason = _messages.EnumField('TransferFailureReasonValueValuesEnum', 15)