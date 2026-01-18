from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadIdentityPool(_messages.Message):
    """Represents a collection of workload identities. You can define IAM
  policies to grant these identities access to Google Cloud resources.

  Enums:
    ModeValueValuesEnum: Immutable. The mode the pool is operating in.
    StateValueValuesEnum: Output only. The state of the pool.

  Fields:
    description: A description of the pool. Cannot exceed 256 characters.
    disabled: Whether the pool is disabled. You cannot use a disabled pool to
      exchange tokens, or use existing tokens to access resources. If the pool
      is re-enabled, existing tokens grant access again.
    displayName: A display name for the pool. Cannot exceed 32 characters.
    expireTime: Output only. Time after which the workload identity pool will
      be permanently purged and cannot be recovered.
    mode: Immutable. The mode the pool is operating in.
    name: Output only. The resource name of the pool.
    sessionDuration: Overrides the lifespan of access tokens issued when
      federating using this pool. If not set, the lifespan of issued access
      tokens is computed based on the type of identity provider: - For AWS
      providers, the default access token lifespan is equal to 15 minutes. -
      For OIDC providers, the default access token lifespan is equal to the
      remaining lifespan of the exchanged OIDC ID token, with a maximum limit
      of 1 hour. If set, session duration must be between 2 minutes and 12
      hours. Organization administrators can further restrict the maximum
      allowed session_duration value using the iam-
      workloadIdentitySessionDuration Resource Setting.
    state: Output only. The state of the pool.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Immutable. The mode the pool is operating in.

    Values:
      MODE_UNSPECIFIED: State unspecified. New pools should not use this mode.
        Pools with an unspecified mode will operate as if they are in
        FEDERATION_ONLY mode.
      FEDERATION_ONLY: FEDERATION_ONLY mode pools can only be used for
        federating external workload identities into Google Cloud. Unless
        otherwise noted, no structure or format constraints are applied to
        workload identities in a FEDERATION_ONLY mode pool, and you may not
        create any resources within the pool besides providers.
      TRUST_DOMAIN: TRUST_DOMAIN mode pools can be used to assign identities
        to either external workloads or those hosted on Google Cloud. All
        identities within a TRUST_DOMAIN mode pool must consist of a single
        namespace and individual workload identifier. The subject identifier
        for all identities must conform to the following format: `ns//sa/`
        WorkloadIdentityPoolProviders cannot be created within TRUST_DOMAIN
        mode pools.
    """
        MODE_UNSPECIFIED = 0
        FEDERATION_ONLY = 1
        TRUST_DOMAIN = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the pool.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The pool is active, and may be used in Google Cloud policies.
      DELETED: The pool is soft-deleted. Soft-deleted pools are permanently
        deleted after approximately 30 days. You can restore a soft-deleted
        pool using UndeleteWorkloadIdentityPool. You cannot reuse the ID of a
        soft-deleted pool until it is permanently deleted. While a pool is
        deleted, you cannot use it to exchange tokens, or use existing tokens
        to access resources. If the pool is undeleted, existing tokens grant
        access again.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2
    description = _messages.StringField(1)
    disabled = _messages.BooleanField(2)
    displayName = _messages.StringField(3)
    expireTime = _messages.StringField(4)
    mode = _messages.EnumField('ModeValueValuesEnum', 5)
    name = _messages.StringField(6)
    sessionDuration = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)