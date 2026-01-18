from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2FindingLimits(_messages.Message):
    """Configuration to control the number of findings returned for inspection.
  This is not used for de-identification or data profiling. When redacting
  sensitive data from images, finding limits don't apply. They can cause
  unexpected or inconsistent results, where only some data is redacted. Don't
  include finding limits in RedactImage requests. Otherwise, Cloud DLP returns
  an error.

  Fields:
    maxFindingsPerInfoType: Configuration of findings limit given for
      specified infoTypes.
    maxFindingsPerItem: Max number of findings that are returned for each item
      scanned. When set within an InspectContentRequest, this field is
      ignored. This value isn't a hard limit. If the number of findings for an
      item reaches this limit, the inspection of that item ends gradually, not
      abruptly. Therefore, the actual number of findings that Cloud DLP
      returns for the item can be multiple times higher than this value.
    maxFindingsPerRequest: Max number of findings that are returned per
      request or job. If you set this field in an InspectContentRequest, the
      resulting maximum value is the value that you set or 3,000, whichever is
      lower. This value isn't a hard limit. If an inspection reaches this
      limit, the inspection ends gradually, not abruptly. Therefore, the
      actual number of findings that Cloud DLP returns can be multiple times
      higher than this value.
  """
    maxFindingsPerInfoType = _messages.MessageField('GooglePrivacyDlpV2InfoTypeLimit', 1, repeated=True)
    maxFindingsPerItem = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    maxFindingsPerRequest = _messages.IntegerField(3, variant=_messages.Variant.INT32)