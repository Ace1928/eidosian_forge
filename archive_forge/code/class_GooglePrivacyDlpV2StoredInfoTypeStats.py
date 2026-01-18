from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StoredInfoTypeStats(_messages.Message):
    """Statistics for a StoredInfoType.

  Fields:
    largeCustomDictionary: StoredInfoType where findings are defined by a
      dictionary of phrases.
  """
    largeCustomDictionary = _messages.MessageField('GooglePrivacyDlpV2LargeCustomDictionaryStats', 1)