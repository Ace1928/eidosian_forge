from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsCreateRequest object.

  Fields:
    parent: Required. The parent resource of the `WasmPlugin` resource. Must
      be in the format `projects/{project}/locations/global`.
    wasmPlugin: A WasmPlugin resource to be passed as the request body.
    wasmPluginId: Required. User-provided ID of the `WasmPlugin` resource to
      be created.
  """
    parent = _messages.StringField(1, required=True)
    wasmPlugin = _messages.MessageField('WasmPlugin', 2)
    wasmPluginId = _messages.StringField(3)