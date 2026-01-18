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
class MainKernelSpecHandler(KernelSpecsAPIHandler):
    """The root kernel spec handler."""

    @web.authenticated
    @authorized
    async def get(self):
        """Get the list of kernel specs."""
        ksm = self.kernel_spec_manager
        km = self.kernel_manager
        model: dict[str, Any] = {}
        model['default'] = km.default_kernel_name
        model['kernelspecs'] = specs = {}
        kspecs = await ensure_async(ksm.get_all_specs())
        for kernel_name, kernel_info in kspecs.items():
            try:
                if is_kernelspec_model(kernel_info):
                    d = kernel_info
                else:
                    d = kernelspec_model(self, kernel_name, kernel_info['spec'], kernel_info['resource_dir'])
            except Exception:
                self.log.error("Failed to load kernel spec: '%s'", kernel_name, exc_info=True)
                continue
            specs[kernel_name] = d
        self.set_header('Content-Type', 'application/json')
        self.finish(json.dumps(model))