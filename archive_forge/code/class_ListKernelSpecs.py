from __future__ import annotations
import errno
import json
import os.path
import sys
import typing as t
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Dict, Instance, List, Unicode
from traitlets.config.application import Application
from . import __version__
from .kernelspec import KernelSpecManager
from .provisioning.factory import KernelProvisionerFactory
class ListKernelSpecs(JupyterApp):
    """An app to list kernel specs."""
    version = __version__
    description = 'List installed kernel specifications.'
    kernel_spec_manager = Instance(KernelSpecManager)
    json_output = Bool(False, help='output spec name and location as machine-readable json.', config=True)
    flags = {'json': ({'ListKernelSpecs': {'json_output': True}}, 'output spec name and location as machine-readable json.'), 'debug': base_flags['debug']}

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(parent=self, data_dir=self.data_dir)

    def start(self) -> dict[str, t.Any] | None:
        """Start the application."""
        paths = self.kernel_spec_manager.find_kernel_specs()
        specs = self.kernel_spec_manager.get_all_specs()
        if not self.json_output:
            if not specs:
                print('No kernels available')
                return None
            name_len = len(sorted(paths, key=lambda name: len(name))[-1])

            def path_key(item: t.Any) -> t.Any:
                """sort key function for Jupyter path priority"""
                path = item[1]
                for idx, prefix in enumerate(self.jupyter_path):
                    if path.startswith(prefix):
                        return (idx, path)
                return (-1, path)
            print('Available kernels:')
            for kernelname, path in sorted(paths.items(), key=path_key):
                print(f'  {kernelname.ljust(name_len)}    {path}')
        else:
            print(json.dumps({'kernelspecs': specs}, indent=2))
        return specs