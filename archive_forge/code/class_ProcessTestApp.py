import atexit
import json
import os
import shutil
import sys
import tempfile
from os import path as osp
from os.path import join as pjoin
from stat import S_IRGRP, S_IROTH, S_IRUSR
from tempfile import TemporaryDirectory
from unittest.mock import patch
import jupyter_core
import jupyterlab_server
from ipykernel.kernelspec import write_kernel_spec
from jupyter_server.serverapp import ServerApp
from jupyterlab_server.process_app import ProcessApp
from traitlets import default
class ProcessTestApp(ProcessApp):
    """A process app for running tests, includes a mock contents directory."""
    allow_origin = '*'

    def initialize_templates(self):
        self.static_paths = [_create_static_dir()]
        self.template_paths = [_create_template_dir()]

    def initialize_settings(self):
        self.env_patch = TestEnv()
        self.env_patch.start()
        ProcessApp.__init__(self)
        self.settings['allow_origin'] = ProcessTestApp.allow_origin
        self.static_dir = self.static_paths[0]
        self.template_dir = self.template_paths[0]
        self.schemas_dir = _create_schemas_dir()
        self.user_settings_dir = _create_user_settings_dir()
        self.workspaces_dir = _create_workspaces_dir()
        self._install_default_kernels()
        self.settings['kernel_manager'].default_kernel_name = 'echo'
        super().initialize_settings()

    def _install_kernel(self, kernel_name, kernel_spec):
        """Install a kernel spec to the data directory.

        Parameters
        ----------
        kernel_name: str
            Name of the kernel.
        kernel_spec: dict
            The kernel spec for the kernel
        """
        paths = jupyter_core.paths
        kernel_dir = pjoin(paths.jupyter_data_dir(), 'kernels', kernel_name)
        os.makedirs(kernel_dir)
        with open(pjoin(kernel_dir, 'kernel.json'), 'w') as f:
            f.write(json.dumps(kernel_spec))

    def _install_default_kernels(self):
        self._install_kernel(kernel_name='echo', kernel_spec={'argv': [sys.executable, '-m', 'jupyterlab.tests.echo_kernel', '-f', '{connection_file}'], 'display_name': 'Echo Kernel', 'language': 'echo'})
        paths = jupyter_core.paths
        ipykernel_dir = pjoin(paths.jupyter_data_dir(), 'kernels', 'ipython')
        write_kernel_spec(ipykernel_dir)

    def _process_finished(self, future):
        self.serverapp.http_server.stop()
        self.serverapp.io_loop.stop()
        self.env_patch.stop()
        try:
            os._exit(future.result())
        except Exception as e:
            self.log.error(str(e))
            os._exit(1)