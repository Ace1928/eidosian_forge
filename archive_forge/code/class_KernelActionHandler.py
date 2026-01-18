import json
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_escape, url_path_join
from ...base.handlers import APIHandler
from .websocket import KernelWebsocketHandler
class KernelActionHandler(KernelsAPIHandler):
    """A kernel action API handler."""

    @web.authenticated
    @authorized
    async def post(self, kernel_id, action):
        """Interrupt or restart a kernel."""
        km = self.kernel_manager
        if action == 'interrupt':
            await ensure_async(km.interrupt_kernel(kernel_id))
            self.set_status(204)
        if action == 'restart':
            try:
                await km.restart_kernel(kernel_id)
            except Exception as e:
                message = 'Exception restarting kernel'
                self.log.error(message, exc_info=True)
                self.write(json.dumps({'message': message, 'traceback': ''}))
                self.set_status(500)
            else:
                model = await ensure_async(km.kernel_model(kernel_id))
                self.write(json.dumps(model, default=json_default))
        self.finish()