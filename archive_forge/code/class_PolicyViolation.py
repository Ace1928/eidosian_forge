from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyViolation(_messages.Message):
    """Returned from an action if one or more policies were violated, and
  therefore the action was prevented. Contains information about what policies
  were violated and why.

  Fields:
    policyViolationDetails: Policy violation details.
  """
    policyViolationDetails = _messages.MessageField('PolicyViolationDetails', 1, repeated=True)