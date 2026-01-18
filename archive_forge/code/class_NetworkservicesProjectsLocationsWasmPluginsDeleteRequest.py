from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsWasmPluginsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsWasmPluginsDeleteRequest object.

  Fields:
    name: Required. A name of the `WasmPlugin` resource to delete. Must be in
      the format
      `projects/{project}/locations/global/wasmPlugins/{wasm_plugin}`.
  """
    name = _messages.StringField(1, required=True)