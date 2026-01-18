from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
class HealthHandler(_SpecialRequestHandler):

    def initialize(self, callback):
        """Initialize the handler

        Parameters
        ----------
        callback : callable
            A function that returns True if the server is healthy

        """
        self._callback = callback

    async def get(self):
        await self.handle_request()

    async def head(self):
        await self.handle_request()

    async def handle_request(self):
        if self.request.uri and '_stcore/' not in self.request.uri:
            new_path = '/_stcore/script-health-check' if 'script-health-check' in self.request.uri else '/_stcore/health'
            emit_endpoint_deprecation_notice(self, new_path=new_path)
        ok, msg = await self._callback()
        if ok:
            self.write(msg)
            self.set_status(200)
            if config.get_option('server.enableXsrfProtection'):
                cookie_kwargs = self.settings.get('xsrf_cookie_kwargs', {})
                self.set_cookie(self.settings.get('xsrf_cookie_name', '_streamlit_xsrf'), self.xsrf_token, **cookie_kwargs)
        else:
            self.set_status(503)
            self.write(msg)