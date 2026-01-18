from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyViolationDetails(_messages.Message):
    """Policy violation details.

  Fields:
    failureMessage: User readable message about why the request violated a
      policy. This is not intended for machine parsing.
    policy: Name of the policy that was violated. Policy resource will be in
      the format of
      `projects/{project}/locations/{location}/policies/{policy}`.
    ruleId: Name of the rule that triggered the policy violation.
  """
    failureMessage = _messages.StringField(1)
    policy = _messages.StringField(2)
    ruleId = _messages.StringField(3)