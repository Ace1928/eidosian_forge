from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DomainMappingCondition(_messages.Message):
    """DomainMappingCondition contains state information for a DomainMapping.

  Fields:
    lastTransitionTime: Last time the condition transitioned from one status
      to another. +optional
    message: Human readable message indicating details about the current
      status. +optional
    reason: One-word CamelCase reason for the condition's current status.
      +optional
    severity: How to interpret failures of this condition, one of Error,
      Warning, Info +optional
    status: Status of the condition, one of True, False, Unknown.
    type: Type of domain mapping condition.
  """
    lastTransitionTime = _messages.StringField(1)
    message = _messages.StringField(2)
    reason = _messages.StringField(3)
    severity = _messages.StringField(4)
    status = _messages.StringField(5)
    type = _messages.StringField(6)