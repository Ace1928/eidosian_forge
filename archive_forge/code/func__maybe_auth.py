import re
import warnings
from typing import Optional, no_type_check
from urllib.parse import urlparse
from tornado import ioloop, web
from tornado.iostream import IOStream
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.utils import JupyterServerAuthWarning
@no_type_check
def _maybe_auth(self):
    """Verify authentication if required.

        Only used when the websocket class does not inherit from JupyterHandler.
        """
    if not self.settings.get('allow_unauthenticated_access', False):
        if not self.request.method:
            raise web.HTTPError(403)
        method = getattr(self, self.request.method.lower())
        if not getattr(method, '__allow_unauthenticated', False):
            user = self.current_user
            if user is None:
                self.log.warning("Couldn't authenticate WebSocket connection")
                raise web.HTTPError(403)