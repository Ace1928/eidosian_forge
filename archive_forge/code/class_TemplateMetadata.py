from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemplateMetadata(_messages.Message):
    """Metadata describing a template.

  Fields:
    defaultStreamingMode: Optional. Indicates the default streaming mode for a
      streaming template. Only valid if both supports_at_least_once and
      supports_exactly_once are true. Possible values: UNSPECIFIED,
      EXACTLY_ONCE and AT_LEAST_ONCE
    description: Optional. A description of the template.
    name: Required. The name of the template.
    parameters: The parameters for the template.
    streaming: Optional. Indicates if the template is streaming or not.
    supportsAtLeastOnce: Optional. Indicates if the streaming template
      supports at least once mode.
    supportsExactlyOnce: Optional. Indicates if the streaming template
      supports exactly once mode.
  """
    defaultStreamingMode = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    parameters = _messages.MessageField('ParameterMetadata', 4, repeated=True)
    streaming = _messages.BooleanField(5)
    supportsAtLeastOnce = _messages.BooleanField(6)
    supportsExactlyOnce = _messages.BooleanField(7)