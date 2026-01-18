from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsVersionsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsVersionsCreateRequest
  object.

  Fields:
    parent: Required. The parent resource of the `WasmPluginVersion` resource.
      Must be in the format
      `projects/{project}/locations/global/wasmPlugins/{wasm_plugin}`.
    wasmPluginVersion: A WasmPluginVersion resource to be passed as the
      request body.
    wasmPluginVersionId: Required. User-provided ID of the `WasmPluginVersion`
      resource to be created.
  """
    parent = _messages.StringField(1, required=True)
    wasmPluginVersion = _messages.MessageField('WasmPluginVersion', 2)
    wasmPluginVersionId = _messages.StringField(3)