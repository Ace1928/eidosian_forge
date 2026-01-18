from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsVersionsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsVersionsDeleteRequest
  object.

  Fields:
    name: Required. A name of the `WasmPluginVersion` resource to delete. Must
      be in the format `projects/{project}/locations/global/wasmPlugins/{wasm_
      plugin}/versions/{wasm_plugin_version}`.
  """
    name = _messages.StringField(1, required=True)