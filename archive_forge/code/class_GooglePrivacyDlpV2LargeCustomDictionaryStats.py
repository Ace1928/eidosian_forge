from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LargeCustomDictionaryStats(_messages.Message):
    """Summary statistics of a custom dictionary.

  Fields:
    approxNumPhrases: Approximate number of distinct phrases in the
      dictionary.
  """
    approxNumPhrases = _messages.IntegerField(1)