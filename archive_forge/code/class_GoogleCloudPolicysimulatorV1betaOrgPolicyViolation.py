from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaOrgPolicyViolation(_messages.Message):
    """OrgPolicyViolation is a resource representing a single resource
  violating a single OrgPolicy constraint.

  Fields:
    customConstraint: The custom constraint being violated.
    error: Any error encountered during the evaluation.
    name: The name of the `OrgPolicyViolation`. Example: organizations/my-
      example-org/locations/global/orgPolicyViolationsPreviews/506a5f7f/orgPol
      icyViolations/38ce`
    resource: The resource violating the constraint.
  """
    customConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2CustomConstraint', 1)
    error = _messages.MessageField('GoogleRpcStatus', 2)
    name = _messages.StringField(3)
    resource = _messages.MessageField('GoogleCloudPolicysimulatorV1betaResourceContext', 4)