from __future__ import annotations
import glob
import json
import os
from typing import Any
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
from ...utils import url_path_join, url_unescape
class KernelSpecsAPIHandler(APIHandler):
    """A kernel spec API handler."""
    auth_resource = AUTH_RESOURCE