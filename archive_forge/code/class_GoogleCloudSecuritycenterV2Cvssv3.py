from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Cvssv3(_messages.Message):
    """Common Vulnerability Scoring System version 3.

  Enums:
    AttackComplexityValueValuesEnum: This metric describes the conditions
      beyond the attacker's control that must exist in order to exploit the
      vulnerability.
    AttackVectorValueValuesEnum: Base Metrics Represents the intrinsic
      characteristics of a vulnerability that are constant over time and
      across user environments. This metric reflects the context by which
      vulnerability exploitation is possible.
    AvailabilityImpactValueValuesEnum: This metric measures the impact to the
      availability of the impacted component resulting from a successfully
      exploited vulnerability.
    ConfidentialityImpactValueValuesEnum: This metric measures the impact to
      the confidentiality of the information resources managed by a software
      component due to a successfully exploited vulnerability.
    IntegrityImpactValueValuesEnum: This metric measures the impact to
      integrity of a successfully exploited vulnerability.
    PrivilegesRequiredValueValuesEnum: This metric describes the level of
      privileges an attacker must possess before successfully exploiting the
      vulnerability.
    ScopeValueValuesEnum: The Scope metric captures whether a vulnerability in
      one vulnerable component impacts resources in components beyond its
      security scope.
    UserInteractionValueValuesEnum: This metric captures the requirement for a
      human user, other than the attacker, to participate in the successful
      compromise of the vulnerable component.

  Fields:
    attackComplexity: This metric describes the conditions beyond the
      attacker's control that must exist in order to exploit the
      vulnerability.
    attackVector: Base Metrics Represents the intrinsic characteristics of a
      vulnerability that are constant over time and across user environments.
      This metric reflects the context by which vulnerability exploitation is
      possible.
    availabilityImpact: This metric measures the impact to the availability of
      the impacted component resulting from a successfully exploited
      vulnerability.
    baseScore: The base score is a function of the base metric scores.
    confidentialityImpact: This metric measures the impact to the
      confidentiality of the information resources managed by a software
      component due to a successfully exploited vulnerability.
    integrityImpact: This metric measures the impact to integrity of a
      successfully exploited vulnerability.
    privilegesRequired: This metric describes the level of privileges an
      attacker must possess before successfully exploiting the vulnerability.
    scope: The Scope metric captures whether a vulnerability in one vulnerable
      component impacts resources in components beyond its security scope.
    userInteraction: This metric captures the requirement for a human user,
      other than the attacker, to participate in the successful compromise of
      the vulnerable component.
  """

    class AttackComplexityValueValuesEnum(_messages.Enum):
        """This metric describes the conditions beyond the attacker's control
    that must exist in order to exploit the vulnerability.

    Values:
      ATTACK_COMPLEXITY_UNSPECIFIED: Invalid value.
      ATTACK_COMPLEXITY_LOW: Specialized access conditions or extenuating
        circumstances do not exist. An attacker can expect repeatable success
        when attacking the vulnerable component.
      ATTACK_COMPLEXITY_HIGH: A successful attack depends on conditions beyond
        the attacker's control. That is, a successful attack cannot be
        accomplished at will, but requires the attacker to invest in some
        measurable amount of effort in preparation or execution against the
        vulnerable component before a successful attack can be expected.
    """
        ATTACK_COMPLEXITY_UNSPECIFIED = 0
        ATTACK_COMPLEXITY_LOW = 1
        ATTACK_COMPLEXITY_HIGH = 2

    class AttackVectorValueValuesEnum(_messages.Enum):
        """Base Metrics Represents the intrinsic characteristics of a
    vulnerability that are constant over time and across user environments.
    This metric reflects the context by which vulnerability exploitation is
    possible.

    Values:
      ATTACK_VECTOR_UNSPECIFIED: Invalid value.
      ATTACK_VECTOR_NETWORK: The vulnerable component is bound to the network
        stack and the set of possible attackers extends beyond the other
        options listed below, up to and including the entire Internet.
      ATTACK_VECTOR_ADJACENT: The vulnerable component is bound to the network
        stack, but the attack is limited at the protocol level to a logically
        adjacent topology.
      ATTACK_VECTOR_LOCAL: The vulnerable component is not bound to the
        network stack and the attacker's path is via read/write/execute
        capabilities.
      ATTACK_VECTOR_PHYSICAL: The attack requires the attacker to physically
        touch or manipulate the vulnerable component.
    """
        ATTACK_VECTOR_UNSPECIFIED = 0
        ATTACK_VECTOR_NETWORK = 1
        ATTACK_VECTOR_ADJACENT = 2
        ATTACK_VECTOR_LOCAL = 3
        ATTACK_VECTOR_PHYSICAL = 4

    class AvailabilityImpactValueValuesEnum(_messages.Enum):
        """This metric measures the impact to the availability of the impacted
    component resulting from a successfully exploited vulnerability.

    Values:
      IMPACT_UNSPECIFIED: Invalid value.
      IMPACT_HIGH: High impact.
      IMPACT_LOW: Low impact.
      IMPACT_NONE: No impact.
    """
        IMPACT_UNSPECIFIED = 0
        IMPACT_HIGH = 1
        IMPACT_LOW = 2
        IMPACT_NONE = 3

    class ConfidentialityImpactValueValuesEnum(_messages.Enum):
        """This metric measures the impact to the confidentiality of the
    information resources managed by a software component due to a
    successfully exploited vulnerability.

    Values:
      IMPACT_UNSPECIFIED: Invalid value.
      IMPACT_HIGH: High impact.
      IMPACT_LOW: Low impact.
      IMPACT_NONE: No impact.
    """
        IMPACT_UNSPECIFIED = 0
        IMPACT_HIGH = 1
        IMPACT_LOW = 2
        IMPACT_NONE = 3

    class IntegrityImpactValueValuesEnum(_messages.Enum):
        """This metric measures the impact to integrity of a successfully
    exploited vulnerability.

    Values:
      IMPACT_UNSPECIFIED: Invalid value.
      IMPACT_HIGH: High impact.
      IMPACT_LOW: Low impact.
      IMPACT_NONE: No impact.
    """
        IMPACT_UNSPECIFIED = 0
        IMPACT_HIGH = 1
        IMPACT_LOW = 2
        IMPACT_NONE = 3

    class PrivilegesRequiredValueValuesEnum(_messages.Enum):
        """This metric describes the level of privileges an attacker must possess
    before successfully exploiting the vulnerability.

    Values:
      PRIVILEGES_REQUIRED_UNSPECIFIED: Invalid value.
      PRIVILEGES_REQUIRED_NONE: The attacker is unauthorized prior to attack,
        and therefore does not require any access to settings or files of the
        vulnerable system to carry out an attack.
      PRIVILEGES_REQUIRED_LOW: The attacker requires privileges that provide
        basic user capabilities that could normally affect only settings and
        files owned by a user. Alternatively, an attacker with Low privileges
        has the ability to access only non-sensitive resources.
      PRIVILEGES_REQUIRED_HIGH: The attacker requires privileges that provide
        significant (e.g., administrative) control over the vulnerable
        component allowing access to component-wide settings and files.
    """
        PRIVILEGES_REQUIRED_UNSPECIFIED = 0
        PRIVILEGES_REQUIRED_NONE = 1
        PRIVILEGES_REQUIRED_LOW = 2
        PRIVILEGES_REQUIRED_HIGH = 3

    class ScopeValueValuesEnum(_messages.Enum):
        """The Scope metric captures whether a vulnerability in one vulnerable
    component impacts resources in components beyond its security scope.

    Values:
      SCOPE_UNSPECIFIED: Invalid value.
      SCOPE_UNCHANGED: An exploited vulnerability can only affect resources
        managed by the same security authority.
      SCOPE_CHANGED: An exploited vulnerability can affect resources beyond
        the security scope managed by the security authority of the vulnerable
        component.
    """
        SCOPE_UNSPECIFIED = 0
        SCOPE_UNCHANGED = 1
        SCOPE_CHANGED = 2

    class UserInteractionValueValuesEnum(_messages.Enum):
        """This metric captures the requirement for a human user, other than the
    attacker, to participate in the successful compromise of the vulnerable
    component.

    Values:
      USER_INTERACTION_UNSPECIFIED: Invalid value.
      USER_INTERACTION_NONE: The vulnerable system can be exploited without
        interaction from any user.
      USER_INTERACTION_REQUIRED: Successful exploitation of this vulnerability
        requires a user to take some action before the vulnerability can be
        exploited.
    """
        USER_INTERACTION_UNSPECIFIED = 0
        USER_INTERACTION_NONE = 1
        USER_INTERACTION_REQUIRED = 2
    attackComplexity = _messages.EnumField('AttackComplexityValueValuesEnum', 1)
    attackVector = _messages.EnumField('AttackVectorValueValuesEnum', 2)
    availabilityImpact = _messages.EnumField('AvailabilityImpactValueValuesEnum', 3)
    baseScore = _messages.FloatField(4)
    confidentialityImpact = _messages.EnumField('ConfidentialityImpactValueValuesEnum', 5)
    integrityImpact = _messages.EnumField('IntegrityImpactValueValuesEnum', 6)
    privilegesRequired = _messages.EnumField('PrivilegesRequiredValueValuesEnum', 7)
    scope = _messages.EnumField('ScopeValueValuesEnum', 8)
    userInteraction = _messages.EnumField('UserInteractionValueValuesEnum', 9)