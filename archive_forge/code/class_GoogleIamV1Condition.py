from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1Condition(_messages.Message):
    """A condition to be met.

  Enums:
    IamValueValuesEnum: Trusted attributes supplied by the IAM system.
    OpValueValuesEnum: An operator to apply the subject with.
    SysValueValuesEnum: Trusted attributes supplied by any service that owns
      resources and uses the IAM system for access control.

  Fields:
    iam: Trusted attributes supplied by the IAM system.
    op: An operator to apply the subject with.
    svc: Trusted attributes discharged by the service.
    sys: Trusted attributes supplied by any service that owns resources and
      uses the IAM system for access control.
    values: The objects of the condition.
  """

    class IamValueValuesEnum(_messages.Enum):
        """Trusted attributes supplied by the IAM system.

    Values:
      NO_ATTR: Default non-attribute.
      AUTHORITY: Either principal or (if present) authority selector.
      ATTRIBUTION: The principal (even if an authority selector is present),
        which must only be used for attribution, not authorization.
      SECURITY_REALM: Any of the security realms in the IAMContext
        (go/security-realms). When used with IN, the condition indicates "any
        of the request's realms match one of the given values; with NOT_IN,
        "none of the realms match any of the given values". Note that a value
        can be: - 'self:campus' (i.e., clients that are in the same campus) -
        'self:metro' (i.e., clients that are in the same metro) - 'self:cloud-
        region' (i.e., allow connections from clients that are in the same
        cloud region) - 'self:prod-region' (i.e., allow connections from
        clients that are in the same prod region) - 'guardians' (i.e., allow
        connections from its guardian realms. See go/security-realms-
        glossary#guardian for more information.) - 'self' [DEPRECATED] (i.e.,
        allow connections from clients that are in the same security realm,
        which is currently but not guaranteed to be campus-sized) - a realm
        (e.g., 'campus-abc') - a realm group (e.g., 'realms-for-borg-cell-xx',
        see: go/realm-groups) A match is determined by a realm group
        membership check performed by a RealmAclRep object (go/realm-acl-
        howto). It is not permitted to grant access based on the *absence* of
        a realm, so realm conditions can only be used in a "positive" context
        (e.g., ALLOW/IN or DENY/NOT_IN).
      APPROVER: An approver (distinct from the requester) that has authorized
        this request. When used with IN, the condition indicates that one of
        the approvers associated with the request matches the specified
        principal, or is a member of the specified group. Approvers can only
        grant additional access, and are thus only used in a strictly positive
        context (e.g. ALLOW/IN or DENY/NOT_IN).
      JUSTIFICATION_TYPE: What types of justifications have been supplied with
        this request. String values should match enum names from
        security.credentials.JustificationType, e.g. "MANUAL_STRING". It is
        not permitted to grant access based on the *absence* of a
        justification, so justification conditions can only be used in a
        "positive" context (e.g., ALLOW/IN or DENY/NOT_IN). Multiple
        justifications, e.g., a Buganizer ID and a manually-entered reason,
        are normal and supported.
      CREDENTIALS_TYPE: What type of credentials have been supplied with this
        request. String values should match enum names from
        security_loas_l2.CredentialsType - currently, only
        CREDS_TYPE_EMERGENCY is supported. It is not permitted to grant access
        based on the *absence* of a credentials type, so the conditions can
        only be used in a "positive" context (e.g., ALLOW/IN or DENY/NOT_IN).
      CREDS_ASSERTION: EXPERIMENTAL -- DO NOT USE. The conditions can only be
        used in a "positive" context (e.g., ALLOW/IN or DENY/NOT_IN).
    """
        NO_ATTR = 0
        AUTHORITY = 1
        ATTRIBUTION = 2
        SECURITY_REALM = 3
        APPROVER = 4
        JUSTIFICATION_TYPE = 5
        CREDENTIALS_TYPE = 6
        CREDS_ASSERTION = 7

    class OpValueValuesEnum(_messages.Enum):
        """An operator to apply the subject with.

    Values:
      NO_OP: Default no-op.
      EQUALS: DEPRECATED. Use IN instead.
      NOT_EQUALS: DEPRECATED. Use NOT_IN instead.
      IN: The condition is true if the subject (or any element of it if it is
        a set) matches any of the supplied values.
      NOT_IN: The condition is true if the subject (or every element of it if
        it is a set) matches none of the supplied values.
      DISCHARGED: Subject is discharged
    """
        NO_OP = 0
        EQUALS = 1
        NOT_EQUALS = 2
        IN = 3
        NOT_IN = 4
        DISCHARGED = 5

    class SysValueValuesEnum(_messages.Enum):
        """Trusted attributes supplied by any service that owns resources and
    uses the IAM system for access control.

    Values:
      NO_ATTR: Default non-attribute type
      REGION: Region of the resource
      SERVICE: Service name
      NAME: Resource name
      IP: IP address of the caller
    """
        NO_ATTR = 0
        REGION = 1
        SERVICE = 2
        NAME = 3
        IP = 4
    iam = _messages.EnumField('IamValueValuesEnum', 1)
    op = _messages.EnumField('OpValueValuesEnum', 2)
    svc = _messages.StringField(3)
    sys = _messages.EnumField('SysValueValuesEnum', 4)
    values = _messages.StringField(5, repeated=True)