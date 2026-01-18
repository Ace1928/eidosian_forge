import json
import os
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server._tz import isoformat, utcfromtimestamp
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler, JupyterHandler
class APIStatusHandler(APIHandler):
    """An API status handler."""
    auth_resource = AUTH_RESOURCE
    _track_activity = False

    @web.authenticated
    @authorized
    async def get(self):
        """Get the API status."""
        started = self.settings.get('started', utcfromtimestamp(0))
        started = isoformat(started)
        kernels = await ensure_async(self.kernel_manager.list_kernels())
        total_connections = sum((k['connections'] for k in kernels))
        last_activity = isoformat(self.application.last_activity())
        model = {'started': started, 'last_activity': last_activity, 'kernels': len(kernels), 'connections': total_connections}
        self.finish(json.dumps(model, sort_keys=True))