from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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