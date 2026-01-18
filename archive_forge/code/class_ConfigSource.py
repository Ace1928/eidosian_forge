from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigSource(_messages.Message):
    """Represents a user-specified configuration for a service (as opposed to
  the the generated service config form provided by `google.api.Service`).
  This is meant to encode service config as manipulated directly by customers,
  rather than the config form resulting from toolchain generation and
  normalization.

  Fields:
    files: Set of source configuration files that are used to generate a
      service config (`google.api.Service`).
    id: A unique ID for a specific instance of this message, typically
      assigned by the client for tracking purpose. If empty, the server may
      choose to generate one instead.
    openApiSpec: OpenAPI specification
    options: Options to cover use of source config within ServiceManager and
      tools
    protoSpec: Protocol buffer API specification
  """
    files = _messages.MessageField('ConfigFile', 1, repeated=True)
    id = _messages.StringField(2)
    openApiSpec = _messages.MessageField('OpenApiSpec', 3)
    options = _messages.MessageField('ConfigOptions', 4)
    protoSpec = _messages.MessageField('ProtoSpec', 5)