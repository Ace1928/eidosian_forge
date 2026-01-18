from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.service_extensions import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def UpdateWasmPlugin(self, name, main_version, update_mask=None, description=None, labels=None, log_config=None):
    """Calls the UpdateWasmPlugin API."""
    request = self.messages.NetworkservicesProjectsLocationsWasmPluginsPatchRequest(name=name, updateMask=update_mask, wasmPlugin=self.messages.WasmPlugin(mainVersionId=main_version, description=description, labels=labels, logConfig=log_config))
    return self._wasm_plugin_client.Patch(request)