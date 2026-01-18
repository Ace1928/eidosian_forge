from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationErrorHandling(_messages.Message):
    """How to handle transformation errors during de-identification. A
  transformation error occurs when the requested transformation is
  incompatible with the data. For example, trying to de-identify an IP address
  using a `DateShift` transformation would result in a transformation error,
  since date info cannot be extracted from an IP address. Information about
  any incompatible transformations, and how they were handled, is returned in
  the response as part of the `TransformationOverviews`.

  Fields:
    leaveUntransformed: Ignore errors
    throwError: Throw an error
  """
    leaveUntransformed = _messages.MessageField('GooglePrivacyDlpV2LeaveUntransformed', 1)
    throwError = _messages.MessageField('GooglePrivacyDlpV2ThrowError', 2)