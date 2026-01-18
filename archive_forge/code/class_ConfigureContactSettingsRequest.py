from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigureContactSettingsRequest(_messages.Message):
    """Request for the `ConfigureContactSettings` method.

  Enums:
    ContactNoticesValueListEntryValuesEnum:

  Fields:
    contactNotices: The list of contact notices that the caller acknowledges.
      The notices needed here depend on the values specified in
      `contact_settings`.
    contactSettings: Fields of the `ContactSettings` to update.
    updateMask: Required. The field mask describing which fields to update as
      a comma-separated list. For example, if only the registrant contact is
      being updated, the `update_mask` is `"registrant_contact"`.
    validateOnly: Validate the request without actually updating the contact
      settings.
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
    contactNotices = _messages.EnumField('ContactNoticesValueListEntryValuesEnum', 1, repeated=True)
    contactSettings = _messages.MessageField('ContactSettings', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)