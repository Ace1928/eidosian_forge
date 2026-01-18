from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpFilterConfig(_messages.Message):
    """HttpFilterConfiguration supplies additional contextual settings for
  networkservices.HttpFilter resources enabled by Traffic Director.

  Fields:
    config: The configuration needed to enable the networkservices.HttpFilter
      resource. The configuration must be YAML formatted and only contain
      fields defined in the protobuf identified in configTypeUrl
    configTypeUrl: The fully qualified versioned proto3 type url of the
      protobuf that the filter expects for its contextual settings, for
      example: type.googleapis.com/google.protobuf.Struct
    filterName: Name of the networkservices.HttpFilter resource this
      configuration belongs to. This name must be known to the xDS client.
      Example: envoy.wasm
  """
    config = _messages.StringField(1)
    configTypeUrl = _messages.StringField(2)
    filterName = _messages.StringField(3)