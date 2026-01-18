from __future__ import annotations
import os
from typing import Final
import tornado.web
from streamlit import config, file_util
from streamlit.logger import get_logger
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
class AddSlashHandler(tornado.web.RequestHandler):

    @tornado.web.addslash
    def get(self):
        pass