from __future__ import annotations
import asyncio
import logging
import mimetypes
import os
import random
import warnings
from typing import Any, Optional, cast
from jupyter_client.session import Session
from tornado import web
from tornado.concurrent import Future
from tornado.escape import json_decode, url_escape, utf8
from tornado.httpclient import HTTPRequest
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.websocket import WebSocketHandler, websocket_connect
from traitlets.config.configurable import LoggingConfigurable
from ..base.handlers import APIHandler, JupyterHandler
from ..utils import url_path_join
from .gateway_client import GatewayClient
from ..services.kernels.handlers import _kernel_id_regex
from ..services.kernelspecs.handlers import kernel_name_regex
class GatewayResourceHandler(APIHandler):
    """Retrieves resources for specific kernelspec definitions from kernel/enterprise gateway."""

    @web.authenticated
    async def get(self, kernel_name, path, include_body=True):
        """Get a gateway resource by name and path."""
        mimetype: Optional[str] = None
        ksm = self.kernel_spec_manager
        kernel_spec_res = await ksm.get_kernel_spec_resource(kernel_name, path)
        if kernel_spec_res is None:
            self.log.warning(f"Kernelspec resource '{path}' for '{kernel_name}' not found.  Gateway may not support resource serving.")
        else:
            mimetype = mimetypes.guess_type(path)[0] or 'text/plain'
        self.finish(kernel_spec_res, set_content_type=mimetype)