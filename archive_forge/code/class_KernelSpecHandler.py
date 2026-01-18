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
class KernelSpecHandler(KernelSpecsAPIHandler):
    """A handler for an individual kernel spec."""

    @web.authenticated
    @authorized
    async def get(self, kernel_name):
        """Get a kernel spec model."""
        ksm = self.kernel_spec_manager
        kernel_name = url_unescape(kernel_name)
        try:
            spec = await ensure_async(ksm.get_kernel_spec(kernel_name))
        except KeyError as e:
            raise web.HTTPError(404, 'Kernel spec %s not found' % kernel_name) from e
        if is_kernelspec_model(spec):
            model = spec
        else:
            model = kernelspec_model(self, kernel_name, spec.to_dict(), spec.resource_dir)
        self.set_header('Content-Type', 'application/json')
        self.finish(json.dumps(model))