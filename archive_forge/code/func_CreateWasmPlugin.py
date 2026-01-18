from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def CreateWasmPlugin(self, name, parent, description=None, labels=None, log_config=None):
    """Calls the CreateWasmPlugin API."""
    request = self.messages.NetworkservicesProjectsLocationsWasmPluginsCreateRequest(parent=parent, wasmPluginId=name, wasmPlugin=self.messages.WasmPlugin(description=description, labels=labels, logConfig=log_config))
    return self._wasm_plugin_client.Create(request)