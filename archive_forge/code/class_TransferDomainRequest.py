from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferDomainRequest(_messages.Message):
    """Deprecated: For more information, see [Cloud Domains feature
  deprecation](https://cloud.google.com/domains/docs/deprecations/feature-
  deprecations). Request for the `TransferDomain` method.

  Enums:
    ContactNoticesValueListEntryValuesEnum:

  Fields:
    authorizationCode: The domain's transfer authorization code. You can
      obtain this from the domain's current registrar.
    contactNotices: The list of contact notices that you acknowledge. The
      notices needed here depend on the values specified in
      `registration.contact_settings`.
    registration: Required. The complete `Registration` resource to be
      created. You can leave `registration.dns_settings` unset to import the
      domain's current DNS configuration from its current registrar. Use this
      option only if you are sure that the domain's current DNS service does
      not cease upon transfer, as is often the case for DNS services provided
      for free by the registrar.
    validateOnly: Validate the request without actually transferring the
      domain.
    yearlyPrice: Required. Acknowledgement of the price to transfer or renew
      the domain for one year. Call `RetrieveTransferParameters` to obtain the
      price, which you must acknowledge.
  """

    class ContactNoticesValueListEntryValuesEnum(_messages.Enum):
        """ContactNoticesValueListEntryValuesEnum enum type.

    Values:
      CONTACT_NOTICE_UNSPECIFIED: The notice is undefined.
      PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT: Required when setting the `privacy`
        field of `ContactSettings` to `PUBLIC_CONTACT_DATA`, which exposes
        contact data publicly.
    """
        CONTACT_NOTICE_UNSPECIFIED = 0
        PUBLIC_CONTACT_DATA_ACKNOWLEDGEMENT = 1
    authorizationCode = _messages.MessageField('AuthorizationCode', 1)
    contactNotices = _messages.EnumField('ContactNoticesValueListEntryValuesEnum', 2, repeated=True)
    registration = _messages.MessageField('Registration', 3)
    validateOnly = _messages.BooleanField(4)
    yearlyPrice = _messages.MessageField('Money', 5)