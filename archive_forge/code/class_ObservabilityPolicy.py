from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObservabilityPolicy(_messages.Message):
    """ObservabilityPolicy is a resource for defining observability parameters.

  Enums:
    ScopeValueValuesEnum: Optional. Scope of the observability policy resource
      indicates its granularity. If not specified, the policy will have no
      effect, unless linked to another resource.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      ObservabilityPolicy resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    labels: Optional. Set of label tags associated with the
      ObservabilityPolicy resource.
    name: Required. Name of the ObservabilityPolicy resource. It matches
      pattern `projects/*/locations/global/observabilityPolicies/`.
    scope: Optional. Scope of the observability policy resource indicates its
      granularity. If not specified, the policy will have no effect, unless
      linked to another resource.
    serviceGraph: Optional. Service graph represents a graph visualization of
      services defined by the user in a scope (e.g in the project).
    updateTime: Output only. The timestamp when the resource was updated.
  """

    class ScopeValueValuesEnum(_messages.Enum):
        """Optional. Scope of the observability policy resource indicates its
    granularity. If not specified, the policy will have no effect, unless
    linked to another resource.

    Values:
      SCOPE_UNSPECIFIED: Default
      PROJECT: The observability policy will be applied at a project level.
    """
        SCOPE_UNSPECIFIED = 0
        PROJECT = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the ObservabilityPolicy
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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    scope = _messages.EnumField('ScopeValueValuesEnum', 5)
    serviceGraph = _messages.MessageField('ServiceGraph', 6)
    updateTime = _messages.StringField(7)