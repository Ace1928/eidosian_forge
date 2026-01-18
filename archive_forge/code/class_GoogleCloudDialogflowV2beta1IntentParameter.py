from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentParameter(_messages.Message):
    """Represents intent parameters.

  Fields:
    defaultValue: Optional. The default value to use when the `value` yields
      an empty result. Default values can be extracted from contexts by using
      the following syntax: `#context_name.parameter_name`.
    displayName: Required. The name of the parameter.
    entityTypeDisplayName: Optional. The name of the entity type, prefixed
      with `@`, that describes values of the parameter. If the parameter is
      required, this must be provided.
    isList: Optional. Indicates whether the parameter represents a list of
      values.
    mandatory: Optional. Indicates whether the parameter is required. That is,
      whether the intent cannot be completed without collecting the parameter
      value.
    name: The unique identifier of this parameter.
    prompts: Optional. The collection of prompts that the agent can present to
      the user in order to collect a value for the parameter.
    value: Optional. The definition of the parameter value. It can be: - a
      constant string, - a parameter value defined as `$parameter_name`, - an
      original parameter value defined as `$parameter_name.original`, - a
      parameter value from some context defined as
      `#context_name.parameter_name`.
  """
    defaultValue = _messages.StringField(1)
    displayName = _messages.StringField(2)
    entityTypeDisplayName = _messages.StringField(3)
    isList = _messages.BooleanField(4)
    mandatory = _messages.BooleanField(5)
    name = _messages.StringField(6)
    prompts = _messages.StringField(7, repeated=True)
    value = _messages.StringField(8)