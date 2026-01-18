from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StoredInfoTypeConfig(_messages.Message):
    """Configuration for stored infoTypes. All fields and subfield are provided
  by the user. For more information, see https://cloud.google.com/sensitive-
  data-protection/docs/creating-custom-infotypes.

  Fields:
    description: Description of the StoredInfoType (max 256 characters).
    dictionary: Store dictionary-based CustomInfoType.
    displayName: Display name of the StoredInfoType (max 256 characters).
    largeCustomDictionary: StoredInfoType where findings are defined by a
      dictionary of phrases.
    regex: Store regular expression-based StoredInfoType.
  """
    description = _messages.StringField(1)
    dictionary = _messages.MessageField('GooglePrivacyDlpV2Dictionary', 2)
    displayName = _messages.StringField(3)
    largeCustomDictionary = _messages.MessageField('GooglePrivacyDlpV2LargeCustomDictionaryConfig', 4)
    regex = _messages.MessageField('GooglePrivacyDlpV2Regex', 5)