from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
class MessageCacheHandler(tornado.web.RequestHandler):
    """Returns ForwardMsgs from our MessageCache"""

    def initialize(self, cache):
        """Initializes the handler.

        Parameters
        ----------
        cache : MessageCache

        """
        self._cache = cache

    def set_default_headers(self):
        if allow_cross_origin_requests():
            self.set_header('Access-Control-Allow-Origin', '*')

    def get(self):
        msg_hash = self.get_argument('hash', None)
        if not config.get_option('global.storeCachedForwardMessagesInMemory'):
            self.set_status(418)
            self.finish()
            return
        if msg_hash is None:
            _LOGGER.error('HTTP request for cached message is missing the hash attribute.')
            self.set_status(404)
            raise tornado.web.Finish()
        message = self._cache.get_message(msg_hash)
        if message is None:
            _LOGGER.error('HTTP request for cached message could not be fulfilled. No such message')
            self.set_status(404)
            raise tornado.web.Finish()
        _LOGGER.debug('MessageCache HIT')
        msg_str = serialize_forward_msg(message)
        self.set_header('Content-Type', 'application/octet-stream')
        self.write(msg_str)
        self.set_status(200)

    def options(self):
        """/OPTIONS handler for preflight CORS checks."""
        self.set_status(204)
        self.finish()