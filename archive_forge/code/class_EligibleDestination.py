from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EligibleDestination(_messages.Message):
    """Message containing the destination details where audit report should be
  uploaded.

  Fields:
    eligibleGcsBucket: Cloud storage bucket location where audit report and
      evidences can be uploaded if specified during the GenerateAuditReport
      API call.
  """
    eligibleGcsBucket = _messages.StringField(1)