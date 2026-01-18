import json
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_escape, url_path_join
from ...base.handlers import APIHandler
from .websocket import KernelWebsocketHandler
class KernelsAPIHandler(APIHandler):
    """A kernels API handler."""
    auth_resource = AUTH_RESOURCE