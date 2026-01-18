from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferParameters(_messages.Message):
    """Deprecated: For more information, see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations). Parameters required to transfer a domain from another
  registrar.

  Enums:
    SupportedPrivacyValueListEntryValuesEnum:
    TransferLockStateValueValuesEnum: Indicates whether the domain is
      protected by a transfer lock. For a transfer to succeed, this must show
      `UNLOCKED`. To unlock a domain, go to its current registrar.

  Fields:
    currentRegistrar: The registrar that currently manages the domain.
    currentRegistrarUri: The URL of the registrar that currently manages the
      domain.
    domainName: The domain name. Unicode domain names are expressed in
      Punycode format.
    nameServers: The name servers that currently store the configuration of
      the domain.
    supportedPrivacy: Contact privacy options that the domain supports.
    transferLockState: Indicates whether the domain is protected by a transfer
      lock. For a transfer to succeed, this must show `UNLOCKED`. To unlock a
      domain, go to its current registrar.
    yearlyPrice: Price to transfer or renew the domain for one year.
  """

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

    class TransferLockStateValueValuesEnum(_messages.Enum):
        """Indicates whether the domain is protected by a transfer lock. For a
    transfer to succeed, this must show `UNLOCKED`. To unlock a domain, go to
    its current registrar.

    Values:
      TRANSFER_LOCK_STATE_UNSPECIFIED: The state is unspecified.
      UNLOCKED: The domain is unlocked and can be transferred to another
        registrar.
      LOCKED: The domain is locked and cannot be transferred to another
        registrar.
    """
        TRANSFER_LOCK_STATE_UNSPECIFIED = 0
        UNLOCKED = 1
        LOCKED = 2
    currentRegistrar = _messages.StringField(1)
    currentRegistrarUri = _messages.StringField(2)
    domainName = _messages.StringField(3)
    nameServers = _messages.StringField(4, repeated=True)
    supportedPrivacy = _messages.EnumField('SupportedPrivacyValueListEntryValuesEnum', 5, repeated=True)
    transferLockState = _messages.EnumField('TransferLockStateValueValuesEnum', 6)
    yearlyPrice = _messages.MessageField('Money', 7)