from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegisterParameters(_messages.Message):
    """Parameters required to register a new domain.

  Enums:
    AvailabilityValueValuesEnum: Indicates whether the domain is available for
      registration. This value is accurate when obtained by calling
      `RetrieveRegisterParameters`, but is approximate when obtained by
      calling `SearchDomains`.
    DomainNoticesValueListEntryValuesEnum:
    SupportedPrivacyValueListEntryValuesEnum:

  Fields:
    availability: Indicates whether the domain is available for registration.
      This value is accurate when obtained by calling
      `RetrieveRegisterParameters`, but is approximate when obtained by
      calling `SearchDomains`.
    domainName: The domain name. Unicode domain names are expressed in
      Punycode format.
    domainNotices: Notices about special properties of the domain.
    supportedPrivacy: Contact privacy options that the domain supports.
    yearlyPrice: Price to register or renew the domain for one year.
  """

    class AvailabilityValueValuesEnum(_messages.Enum):
        """Indicates whether the domain is available for registration. This value
    is accurate when obtained by calling `RetrieveRegisterParameters`, but is
    approximate when obtained by calling `SearchDomains`.

    Values:
      AVAILABILITY_UNSPECIFIED: The availability is unspecified.
      AVAILABLE: The domain is available for registration.
      UNAVAILABLE: The domain is not available for registration. Generally
        this means it is already registered to another party.
      UNSUPPORTED: The domain is not currently supported by Cloud Domains, but
        may be available elsewhere.
      UNKNOWN: Cloud Domains is unable to determine domain availability,
        generally due to system maintenance at the domain name registry.
    """
        AVAILABILITY_UNSPECIFIED = 0
        AVAILABLE = 1
        UNAVAILABLE = 2
        UNSUPPORTED = 3
        UNKNOWN = 4

    class DomainNoticesValueListEntryValuesEnum(_messages.Enum):
        """DomainNoticesValueListEntryValuesEnum enum type.

    Values:
      DOMAIN_NOTICE_UNSPECIFIED: The notice is undefined.
      HSTS_PRELOADED: Indicates that the domain is preloaded on the HTTP
        Strict Transport Security list in browsers. Serving a website on such
        domain requires an SSL certificate. For details, see [how to get an
        SSL certificate](https://support.google.com/domains/answer/7638036).
    """
        DOMAIN_NOTICE_UNSPECIFIED = 0
        HSTS_PRELOADED = 1

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
    availability = _messages.EnumField('AvailabilityValueValuesEnum', 1)
    domainName = _messages.StringField(2)
    domainNotices = _messages.EnumField('DomainNoticesValueListEntryValuesEnum', 3, repeated=True)
    supportedPrivacy = _messages.EnumField('SupportedPrivacyValueListEntryValuesEnum', 4, repeated=True)
    yearlyPrice = _messages.MessageField('Money', 5)