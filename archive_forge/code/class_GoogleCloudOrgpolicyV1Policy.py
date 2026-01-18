from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudOrgpolicyV1Policy(_messages.Message):
    """Defines a Cloud Organization `Policy` which is used to specify
  `Constraints` for configurations of Cloud Platform resources.

  Fields:
    booleanPolicy: For boolean `Constraints`, whether to enforce the
      `Constraint` or not.
    constraint: The name of the `Constraint` the `Policy` is configuring, for
      example, `constraints/serviceuser.services`. A [list of available
      constraints](/resource-manager/docs/organization-policy/org-policy-
      constraints) is available. Immutable after creation.
    etag: An opaque tag indicating the current version of the `Policy`, used
      for concurrency control. When the `Policy` is returned from either a
      `GetPolicy` or a `ListOrgPolicy` request, this `etag` indicates the
      version of the current `Policy` to use when executing a read-modify-
      write loop. When the `Policy` is returned from a `GetEffectivePolicy`
      request, the `etag` will be unset. When the `Policy` is used in a
      `SetOrgPolicy` method, use the `etag` value that was returned from a
      `GetOrgPolicy` request as part of a read-modify-write loop for
      concurrency control. Not setting the `etag`in a `SetOrgPolicy` request
      will result in an unconditional write of the `Policy`.
    listPolicy: List of values either allowed or disallowed.
    restoreDefault: Restores the default behavior of the constraint;
      independent of `Constraint` type.
    updateTime: The time stamp the `Policy` was previously updated. This is
      set by the server, not specified by the caller, and represents the last
      time a call to `SetOrgPolicy` was made for that `Policy`. Any value set
      by the client will be ignored.
    version: Version of the `Policy`. Default version is 0;
  """
    booleanPolicy = _messages.MessageField('GoogleCloudOrgpolicyV1BooleanPolicy', 1)
    constraint = _messages.StringField(2)
    etag = _messages.BytesField(3)
    listPolicy = _messages.MessageField('GoogleCloudOrgpolicyV1ListPolicy', 4)
    restoreDefault = _messages.MessageField('GoogleCloudOrgpolicyV1RestoreDefault', 5)
    updateTime = _messages.StringField(6)
    version = _messages.IntegerField(7, variant=_messages.Variant.INT32)