from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2LDiversityConfig(_messages.Message):
    """l-diversity metric, used for analysis of reidentification risk.

  Fields:
    quasiIds: Set of quasi-identifiers indicating how equivalence classes are
      defined for the l-diversity computation. When multiple fields are
      specified, they are considered a single composite key.
    sensitiveAttribute: Sensitive field for computing the l-value.
  """
    quasiIds = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1, repeated=True)
    sensitiveAttribute = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2)