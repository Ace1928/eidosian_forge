from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetAssociation(_messages.Message):
    """Message describing TargetAssociation object

  Enums:
    EnableAuthorizationDebugLogValueValuesEnum: Optional. Enable the
      generation of authorization debug logs for the target.

  Fields:
    asmWorkload: Immutable. AnthosServiceMesh based workload. Authorization
      Toolkit does not auto configure the authorization settings on the
      workload.
    createTime: Output only. [Output only] Create time stamp
    displayName: Optional. An arbitrary user-provided name for
      TargetAssociation. The display name should adhere to the following
      format. * Must be 6 to 63 characters in length. * Can only contain
      lowercase letters, numbers, and hyphens. * Must start with a letter.
    enableAuthorizationAuditLog: Optional. Enable the generation of
      authorization audit logs for the target.
    enableAuthorizationDebugLog: Optional. Enable the generation of
      authorization debug logs for the target.
    name: Identifier. name of resource
    policies: Optional. List of policies with full policy name and its
      configuration
    updateTime: Output only. [Output only] Update time stamp
  """

    class EnableAuthorizationDebugLogValueValuesEnum(_messages.Enum):
        """Optional. Enable the generation of authorization debug logs for the
    target.

    Values:
      LOG_NONE: Disable the authorization debug log.
      LOG_ERROR: Generate authorization debug log only in case the
        authorization result is an error.
      LOG_DENY_AND_ERROR: Generate authorization debug log only in case the
        authorization is denied or the authorization result is an error.
      LOG_ALL: Generate authorization debug log for all the authorization
        results.
    """
        LOG_NONE = 0
        LOG_ERROR = 1
        LOG_DENY_AND_ERROR = 2
        LOG_ALL = 3
    asmWorkload = _messages.MessageField('AnthosServiceMesh', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    enableAuthorizationAuditLog = _messages.BooleanField(4)
    enableAuthorizationDebugLog = _messages.EnumField('EnableAuthorizationDebugLogValueValuesEnum', 5)
    name = _messages.StringField(6)
    policies = _messages.MessageField('PolicyConfig', 7, repeated=True)
    updateTime = _messages.StringField(8)