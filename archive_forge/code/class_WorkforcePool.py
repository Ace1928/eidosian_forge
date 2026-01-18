from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforcePool(_messages.Message):
    """Represents a collection of external workforces. Provides namespaces for
  federated users that can be referenced in IAM policies.

  Enums:
    StateValueValuesEnum: Output only. The state of the pool.

  Fields:
    accessRestrictions: Optional. Configure access restrictions on the
      workforce pool users. This is an optional field. If specified web sign-
      in can be restricted to given set of services or programmatic sign-in
      can be disabled for pool users.
    description: A user-specified description of the pool. Cannot exceed 256
      characters.
    disabled: Disables the workforce pool. You cannot use a disabled pool to
      exchange tokens, or use existing tokens to access resources. If the pool
      is re-enabled, existing tokens grant access again.
    displayName: A user-specified display name of the pool in Google Cloud
      Console. Cannot exceed 32 characters.
    expireTime: Output only. Time after which the workforce pool will be
      permanently purged and cannot be recovered.
    name: Output only. The resource name of the pool. Format:
      `locations/{location}/workforcePools/{workforce_pool_id}`
    parent: Immutable. The resource name of the parent. Format:
      `organizations/{org-id}`.
    sessionDuration: Duration that the Google Cloud access tokens, console
      sign-in sessions, and `gcloud` sign-in sessions from this pool are
      valid. Must be greater than 15 minutes (900s) and less than 12 hours
      (43200s). If `session_duration` is not configured, minted credentials
      have a default duration of one hour (3600s). For SAML providers, the
      lifetime of the token is the minimum of the `session_duration` and the
      `SessionNotOnOrAfter` claim in the SAML assertion.
    state: Output only. The state of the pool.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the pool.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The pool is active and may be used in Google Cloud policies.
      DELETED: The pool is soft-deleted. Soft-deleted pools are permanently
        deleted after approximately 30 days. You can restore a soft-deleted
        pool using UndeleteWorkforcePool. You cannot reuse the ID of a soft-
        deleted pool until it is permanently deleted. While a pool is deleted,
        you cannot use it to exchange tokens, or use existing tokens to access
        resources. If the pool is undeleted, existing tokens grant access
        again.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    accessRestrictions = _messages.MessageField('AccessRestrictions', 1)
    description = _messages.StringField(2)
    disabled = _messages.BooleanField(3)
    displayName = _messages.StringField(4)
    expireTime = _messages.StringField(5)
    name = _messages.StringField(6)
    parent = _messages.StringField(7)
    sessionDuration = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)