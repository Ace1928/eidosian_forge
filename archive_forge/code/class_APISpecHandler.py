import json
import os
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server._tz import isoformat, utcfromtimestamp
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler, JupyterHandler
class APISpecHandler(web.StaticFileHandler, JupyterHandler):
    """A spec handler for the REST API."""
    auth_resource = AUTH_RESOURCE

    def initialize(self):
        """Initialize the API spec handler."""
        web.StaticFileHandler.initialize(self, path=os.path.dirname(__file__))

    @web.authenticated
    @authorized
    def head(self):
        return self.get('api.yaml', include_body=False)

    @web.authenticated
    @authorized
    def get(self):
        """Get the API spec."""
        self.log.warning('Serving api spec (experimental, incomplete)')
        return web.StaticFileHandler.get(self, 'api.yaml')

    def get_content_type(self):
        """Get the content type."""
        return 'text/x-yaml'