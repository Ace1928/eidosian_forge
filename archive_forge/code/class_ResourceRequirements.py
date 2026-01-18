from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceRequirements(_messages.Message):
    """ResourceRequirements describes the compute resource requirements.

  Messages:
    LimitsValue: Limits describes the maximum amount of compute resources
      allowed. Only 'cpu' and 'memory' keys are supported. * For supported
      'cpu' values, go to https://cloud.google.com/run/docs/configuring/cpu. *
      For supported 'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits
    RequestsValue: Requests describes the minimum amount of compute resources
      required. Only `cpu` and `memory` are supported. If Requests is omitted
      for a container, it defaults to Limits if that is explicitly specified,
      otherwise to an implementation-defined value. * For supported 'cpu'
      values, go to https://cloud.google.com/run/docs/configuring/cpu. * For
      supported 'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits

  Fields:
    limits: Limits describes the maximum amount of compute resources allowed.
      Only 'cpu' and 'memory' keys are supported. * For supported 'cpu'
      values, go to https://cloud.google.com/run/docs/configuring/cpu. * For
      supported 'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits
    requests: Requests describes the minimum amount of compute resources
      required. Only `cpu` and `memory` are supported. If Requests is omitted
      for a container, it defaults to Limits if that is explicitly specified,
      otherwise to an implementation-defined value. * For supported 'cpu'
      values, go to https://cloud.google.com/run/docs/configuring/cpu. * For
      supported 'memory' values and syntax, go to
      https://cloud.google.com/run/docs/configuring/memory-limits
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LimitsValue(_messages.Message):
        """Limits describes the maximum amount of compute resources allowed. Only
    'cpu' and 'memory' keys are supported. * For supported 'cpu' values, go to
    https://cloud.google.com/run/docs/configuring/cpu. * For supported
    'memory' values and syntax, go to
    https://cloud.google.com/run/docs/configuring/memory-limits

    Messages:
      AdditionalProperty: An additional property for a LimitsValue object.

    Fields:
      additionalProperties: Additional properties of type LimitsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LimitsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestsValue(_messages.Message):
        """Requests describes the minimum amount of compute resources required.
    Only `cpu` and `memory` are supported. If Requests is omitted for a
    container, it defaults to Limits if that is explicitly specified,
    otherwise to an implementation-defined value. * For supported 'cpu'
    values, go to https://cloud.google.com/run/docs/configuring/cpu. * For
    supported 'memory' values and syntax, go to
    https://cloud.google.com/run/docs/configuring/memory-limits

    Messages:
      AdditionalProperty: An additional property for a RequestsValue object.

    Fields:
      additionalProperties: Additional properties of type RequestsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    limits = _messages.MessageField('LimitsValue', 1)
    requests = _messages.MessageField('RequestsValue', 2)