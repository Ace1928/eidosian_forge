from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthorizationPolicy(_messages.Message):
    """AuthorizationPolicy is a resource that specifies how a server should
  authorize incoming connections. This resource in itself does not change the
  configuration unless it's attached to a target https proxy or endpoint
  config selector resource.

  Enums:
    ActionValueValuesEnum: Required. The action to take when a rule match is
      found. Possible values are "ALLOW" or "DENY".

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      AuthorizationPolicy resource.

  Fields:
    action: Required. The action to take when a rule match is found. Possible
      values are "ALLOW" or "DENY".
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. Free-text description of the resource.
    internalCaller: Optional. A flag set to identify internal controllers
      Setting this will trigger a P4SA check to validate the caller is from an
      allowlisted service's P4SA even if other optional fields are unset.
    labels: Optional. Set of label tags associated with the
      AuthorizationPolicy resource.
    name: Required. Name of the AuthorizationPolicy resource. It matches
      pattern
      `projects/{project}/locations/{location}/authorizationPolicies/`.
    rules: Optional. List of rules to match. Note that at least one of the
      rules must match in order for the action specified in the 'action' field
      to be taken. A rule is a match if there is a matching source and
      destination. If left blank, the action specified in the `action` field
      will be applied on every request.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Required. The action to take when a rule match is found. Possible
    values are "ALLOW" or "DENY".

    Values:
      ACTION_UNSPECIFIED: Default value.
      ALLOW: Grant access.
      DENY: Deny access. Deny rules should be avoided unless they are used to
        provide a default "deny all" fallback.
    """
        ACTION_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the AuthorizationPolicy
    resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    internalCaller = _messages.BooleanField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    rules = _messages.MessageField('Rule', 7, repeated=True)
    updateTime = _messages.StringField(8)