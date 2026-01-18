from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaAccessBoundaryPolicy(_messages.Message):
    """Policy details for principal access boundary policy, a policy type that
  define access boundary for principal sets

  Fields:
    rules: Required. A list of rules that specify the behavior of the
      `Policy`. The list is limited to 5 rules.
    version: Optional. The type of versioning that will be enforced on the
      policy. If no version is specified, policies will default to use
      latest_version.
  """
    rules = _messages.MessageField('GoogleIamV3betaAccessBoundaryPolicyRule', 1, repeated=True)
    version = _messages.MessageField('GoogleIamV3betaAccessBoundaryPolicyEnforcementVersion', 2)