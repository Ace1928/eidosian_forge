import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
class NotebooksRedirectHandler(JupyterHandler):
    """Redirect /api/notebooks to /api/contents"""
    SUPPORTED_METHODS = ('GET', 'PUT', 'PATCH', 'POST', 'DELETE')

    @allow_unauthenticated
    def get(self, path):
        """Handle a notebooks redirect."""
        self.log.warning('/api/notebooks is deprecated, use /api/contents')
        self.redirect(url_path_join(self.base_url, 'api/contents', url_escape(path)))
    put = patch = post = delete = get