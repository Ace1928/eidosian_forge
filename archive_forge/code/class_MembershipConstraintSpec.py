from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintSpec(_messages.Message):
    """The spec defining this constraint. See https://open-policy-
  agent.github.io/gatekeeper/website/docs/howto#constraints.

  Enums:
    EnforcementActionValueValuesEnum: spec.enforcement_action.

  Fields:
    enforcementAction: spec.enforcement_action.
    kubernetesMatch: Reserved: The match specified against GCP resources.
      GCPMatch gcp_match = 3;
    parameters: The parameters a constraint expects.
  """

    class EnforcementActionValueValuesEnum(_messages.Enum):
        """spec.enforcement_action.

    Values:
      ENFORCEMENT_ACTION_UNSPECIFIED: Unspecified state for an enforcement
        action.
      ENFORCEMENT_ACTION_DENY: The resource is denied admission to the
        membership.
      ENFORCEMENT_ACTION_DRYRUN: Allows testing constraints without enforcing
        them.
      ENFORCEMENT_ACTION_WARN: Provides immediate feedback on why a resource
        violates a constraint.
    """
        ENFORCEMENT_ACTION_UNSPECIFIED = 0
        ENFORCEMENT_ACTION_DENY = 1
        ENFORCEMENT_ACTION_DRYRUN = 2
        ENFORCEMENT_ACTION_WARN = 3
    enforcementAction = _messages.EnumField('EnforcementActionValueValuesEnum', 1)
    kubernetesMatch = _messages.MessageField('KubernetesMatch', 2)
    parameters = _messages.MessageField('MembershipConstraintSpecParameters', 3)