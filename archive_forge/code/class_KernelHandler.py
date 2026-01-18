import json
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_escape, url_path_join
from ...base.handlers import APIHandler
from .websocket import KernelWebsocketHandler
class KernelHandler(KernelsAPIHandler):
    """A kernel API handler."""

    @web.authenticated
    @authorized
    async def get(self, kernel_id):
        """Get a kernel model."""
        km = self.kernel_manager
        model = await ensure_async(km.kernel_model(kernel_id))
        self.finish(json.dumps(model, default=json_default))

    @web.authenticated
    @authorized
    async def delete(self, kernel_id):
        """Remove a kernel."""
        km = self.kernel_manager
        await ensure_async(km.shutdown_kernel(kernel_id))
        self.set_status(204)
        self.finish()