from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaOrgPolicyOverlay(_messages.Message):
    """The proposed changes to OrgPolicy.

  Fields:
    customConstraints: Optional. The OrgPolicy CustomConstraint changes to
      preview violations for. Any existing CustomConstraints with the same
      name will be overridden in the simulation. That is, violations will be
      determined as if all custom constraints in the overlay were
      instantiated. Only a single custom_constraint is supported in the
      overlay at a time. For evaluating multiple constraints, multiple
      `GenerateOrgPolicyViolationsPreview` requests are made, where each
      request evaluates a single constraint.
    policies: Optional. The OrgPolicy changes to preview violations for. Any
      existing OrgPolicies with the same name will be overridden in the
      simulation. That is, violations will be determined as if all policies in
      the overlay were created or updated.
  """
    customConstraints = _messages.MessageField('GoogleCloudPolicysimulatorV1betaOrgPolicyOverlayCustomConstraintOverlay', 1, repeated=True)
    policies = _messages.MessageField('GoogleCloudPolicysimulatorV1betaOrgPolicyOverlayPolicyOverlay', 2, repeated=True)