from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2Constraint(_messages.Message):
    """A constraint describes a way to restrict resource's configuration. For
  example, you could enforce a constraint that controls which Google Cloud
  services can be activated across an organization, or whether a Compute
  Engine instance can have serial port connections established. Constraints
  can be configured by the organization policy administrator to fit the needs
  of the organization by setting a policy that includes constraints at
  different locations in the organization's resource hierarchy. Policies are
  inherited down the resource hierarchy from higher levels, but can also be
  overridden. For details about the inheritance rules please read about
  `policies`. Constraints have a default behavior determined by the
  `constraint_default` field, which is the enforcement behavior that is used
  in the absence of a policy being defined or inherited for the resource in
  question.

  Enums:
    ConstraintDefaultValueValuesEnum: The evaluation behavior of this
      constraint in the absence of a policy.

  Fields:
    booleanConstraint: Defines this constraint as being a BooleanConstraint.
    constraintDefault: The evaluation behavior of this constraint in the
      absence of a policy.
    customConstraint: Defines this constraint as being a CustomConstraint.
    description: Detailed description of what this constraint controls as well
      as how and where it is enforced. Mutable.
    displayName: The human readable name. Mutable.
    listConstraint: Defines this constraint as being a ListConstraint.
    name: Immutable. The resource name of the constraint. Must be in one of
      the following forms: *
      `projects/{project_number}/constraints/{constraint_name}` *
      `folders/{folder_id}/constraints/{constraint_name}` *
      `organizations/{organization_id}/constraints/{constraint_name}` For
      example, "/projects/123/constraints/compute.disableSerialPortAccess".
    supportsDryRun: Shows if dry run is supported for this constraint or not.
  """

    class ConstraintDefaultValueValuesEnum(_messages.Enum):
        """The evaluation behavior of this constraint in the absence of a policy.

    Values:
      CONSTRAINT_DEFAULT_UNSPECIFIED: This is only used for distinguishing
        unset values and should never be used.
      ALLOW: Indicate that all values are allowed for list constraints.
        Indicate that enforcement is off for boolean constraints.
      DENY: Indicate that all values are denied for list constraints. Indicate
        that enforcement is on for boolean constraints.
    """
        CONSTRAINT_DEFAULT_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2
    booleanConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2ConstraintBooleanConstraint', 1)
    constraintDefault = _messages.EnumField('ConstraintDefaultValueValuesEnum', 2)
    customConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2ConstraintGoogleDefinedCustomConstraint', 3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    listConstraint = _messages.MessageField('GoogleCloudOrgpolicyV2ConstraintListConstraint', 6)
    name = _messages.StringField(7)
    supportsDryRun = _messages.BooleanField(8)