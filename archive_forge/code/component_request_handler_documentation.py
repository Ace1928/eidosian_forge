from __future__ import annotations
import mimetypes
import os
from typing import Final
import tornado.web
import streamlit.web.server.routes
from streamlit.components.v1.components import ComponentRegistry
from streamlit.logger import get_logger
Return the URL for a component file with the given ID.