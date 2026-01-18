from __future__ import annotations
import mimetypes
import os
from typing import Final
import tornado.web
import streamlit.web.server.routes
from streamlit.components.v1.components import ComponentRegistry
from streamlit.logger import get_logger
class ComponentRequestHandler(tornado.web.RequestHandler):

    def initialize(self, registry: ComponentRegistry):
        self._registry = registry

    def get(self, path: str) -> None:
        parts = path.split('/')
        component_name = parts[0]
        component_root = self._registry.get_component_path(component_name)
        if component_root is None:
            self.write('not found')
            self.set_status(404)
            return
        component_root = os.path.realpath(component_root)
        filename = '/'.join(parts[1:])
        abspath = os.path.realpath(os.path.join(component_root, filename))
        if os.path.commonpath([component_root, abspath]) != component_root:
            self.write('forbidden')
            self.set_status(403)
            return
        try:
            with open(abspath, 'rb') as file:
                contents = file.read()
        except OSError as e:
            _LOGGER.error('ComponentRequestHandler: GET %s read error', abspath, exc_info=e)
            self.write('read error')
            self.set_status(404)
            return
        self.write(contents)
        self.set_header('Content-Type', self.get_content_type(abspath))
        self.set_extra_headers(path)

    def set_extra_headers(self, path: str) -> None:
        """Disable cache for HTML files.

        Other assets like JS and CSS are suffixed with their hash, so they can
        be cached indefinitely.
        """
        is_index_url = len(path) == 0
        if is_index_url or path.endswith('.html'):
            self.set_header('Cache-Control', 'no-cache')
        else:
            self.set_header('Cache-Control', 'public')

    def set_default_headers(self) -> None:
        if streamlit.web.server.routes.allow_cross_origin_requests():
            self.set_header('Access-Control-Allow-Origin', '*')

    def options(self) -> None:
        """/OPTIONS handler for preflight CORS checks."""
        self.set_status(204)
        self.finish()

    @staticmethod
    def get_content_type(abspath: str) -> str:
        """Returns the ``Content-Type`` header to be used for this request.
        From tornado.web.StaticFileHandler.
        """
        mime_type, encoding = mimetypes.guess_type(abspath)
        if encoding == 'gzip':
            return 'application/gzip'
        elif encoding is not None:
            return 'application/octet-stream'
        elif mime_type is not None:
            return mime_type
        else:
            return 'application/octet-stream'

    @staticmethod
    def get_url(file_id: str) -> str:
        """Return the URL for a component file with the given ID."""
        return f'components/{file_id}'