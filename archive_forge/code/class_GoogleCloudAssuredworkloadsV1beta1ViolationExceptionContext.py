from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1ViolationExceptionContext(_messages.Message):
    """Violation exception detail.

  Fields:
    acknowledgementTime: Timestamp when the violation was acknowledged.
    comment: Business justification provided towards the acknowledgement of
      the violation.
    userName: Name of the user (or service account) who acknowledged the
      violation.
  """
    acknowledgementTime = _messages.StringField(1)
    comment = _messages.StringField(2)
    userName = _messages.StringField(3)