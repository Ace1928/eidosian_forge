from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
class StaticFileHandler(tornado.web.StaticFileHandler):

    def initialize(self, path, default_filename, get_pages):
        self._pages = get_pages()
        super().initialize(path=path, default_filename=default_filename)

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

    def parse_url_path(self, url_path: str) -> str:
        url_parts = url_path.split('/')
        maybe_page_name = url_parts[0]
        if maybe_page_name in self._pages:
            if len(url_parts) == 1:
                return 'index.html'
            url_path = '/'.join(url_parts[1:])
        return super().parse_url_path(url_path)

    def write_error(self, status_code: int, **kwargs) -> None:
        if status_code == 404:
            index_file = os.path.join(file_util.get_static_dir(), 'index.html')
            self.render(index_file)
        else:
            super().write_error(status_code, **kwargs)