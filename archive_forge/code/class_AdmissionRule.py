from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AdmissionRule(_messages.Message):
    """An admission rule specifies either that all container images used in a
  pod creation request must be attested to by one or more attestors, that all
  pod creations will be allowed, or that all pod creations will be denied.
  Images matching an admission allowlist pattern are exempted from admission
  rules and will never block a pod creation.

  Enums:
    EnforcementModeValueValuesEnum: Required. The action when a pod creation
      is denied by the admission rule.
    EvaluationModeValueValuesEnum: Required. How this admission rule will be
      evaluated.

  Fields:
    enforcementMode: Required. The action when a pod creation is denied by the
      admission rule.
    evaluationMode: Required. How this admission rule will be evaluated.
    requireAttestationsBy: Optional. The resource names of the attestors that
      must attest to a container image, in the format
      `projects/*/attestors/*`. Each attestor must exist before a policy can
      reference it. To add an attestor to a policy the principal issuing the
      policy change request must be able to read the attestor resource. Note:
      this field must be non-empty when the evaluation_mode field specifies
      REQUIRE_ATTESTATION, otherwise it must be empty.
  """

    class EnforcementModeValueValuesEnum(_messages.Enum):
        """Required. The action when a pod creation is denied by the admission
    rule.

    Values:
      ENFORCEMENT_MODE_UNSPECIFIED: Do not use.
      ENFORCED_BLOCK_AND_AUDIT_LOG: Enforce the admission rule by blocking the
        pod creation.
      DRYRUN_AUDIT_LOG_ONLY: Dryrun mode: Audit logging only. This will allow
        the pod creation as if the admission request had specified break-
        glass.
    """
        ENFORCEMENT_MODE_UNSPECIFIED = 0
        ENFORCED_BLOCK_AND_AUDIT_LOG = 1
        DRYRUN_AUDIT_LOG_ONLY = 2

    class EvaluationModeValueValuesEnum(_messages.Enum):
        """Required. How this admission rule will be evaluated.

    Values:
      EVALUATION_MODE_UNSPECIFIED: Do not use.
      ALWAYS_ALLOW: This rule allows all all pod creations.
      REQUIRE_ATTESTATION: This rule allows a pod creation if all the
        attestors listed in `require_attestations_by` have valid attestations
        for all of the images in the pod spec.
      ALWAYS_DENY: This rule denies all pod creations.
    """
        EVALUATION_MODE_UNSPECIFIED = 0
        ALWAYS_ALLOW = 1
        REQUIRE_ATTESTATION = 2
        ALWAYS_DENY = 3
    enforcementMode = _messages.EnumField('EnforcementModeValueValuesEnum', 1)
    evaluationMode = _messages.EnumField('EvaluationModeValueValuesEnum', 2)
    requireAttestationsBy = _messages.StringField(3, repeated=True)