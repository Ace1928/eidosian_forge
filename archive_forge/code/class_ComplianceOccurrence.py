from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceOccurrence(_messages.Message):
    """An indication that the compliance checks in the associated
  ComplianceNote were not satisfied for particular resources or a specified
  reason.

  Fields:
    nonComplianceReason: A string attribute.
    nonCompliantFiles: A NonCompliantFile attribute.
  """
    nonComplianceReason = _messages.StringField(1)
    nonCompliantFiles = _messages.MessageField('NonCompliantFile', 2, repeated=True)