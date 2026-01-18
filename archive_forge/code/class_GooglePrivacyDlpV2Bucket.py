from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Bucket(_messages.Message):
    """Bucket is represented as a range, along with replacement values.

  Fields:
    max: Upper bound of the range, exclusive; type must match min.
    min: Lower bound of the range, inclusive. Type should be the same as max
      if used.
    replacementValue: Required. Replacement value for this bucket.
  """
    max = _messages.MessageField('GooglePrivacyDlpV2Value', 1)
    min = _messages.MessageField('GooglePrivacyDlpV2Value', 2)
    replacementValue = _messages.MessageField('GooglePrivacyDlpV2Value', 3)