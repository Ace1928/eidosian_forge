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
class InstallKernelSpec(JupyterApp):
    """An app to install a kernel spec."""
    version = __version__
    description = 'Install a kernel specification directory.\n\n    Given a SOURCE DIRECTORY containing a kernel spec,\n    jupyter will copy that directory into one of the Jupyter kernel directories.\n    The default is to install kernelspecs for all users.\n    `--user` can be specified to install a kernel only for the current user.\n    '
    examples = '\n    jupyter kernelspec install /path/to/my_kernel --user\n    '
    usage = 'jupyter kernelspec install SOURCE_DIR [--options]'
    kernel_spec_manager = Instance(KernelSpecManager)

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(data_dir=self.data_dir)
    sourcedir = Unicode()
    kernel_name = Unicode('', config=True, help='Install the kernel spec with this name')

    def _kernel_name_default(self) -> str:
        return os.path.basename(self.sourcedir)
    user = Bool(False, config=True, help='\n        Try to install the kernel spec to the per-user directory instead of\n        the system or environment directory.\n        ')
    prefix = Unicode('', config=True, help='Specify a prefix to install to, e.g. an env.\n        The kernelspec will be installed in PREFIX/share/jupyter/kernels/\n        ')
    replace = Bool(False, config=True, help='Replace any existing kernel spec with this name.')
    aliases = {'name': 'InstallKernelSpec.kernel_name', 'prefix': 'InstallKernelSpec.prefix'}
    aliases.update(base_aliases)
    flags = {'user': ({'InstallKernelSpec': {'user': True}}, 'Install to the per-user kernel registry'), 'replace': ({'InstallKernelSpec': {'replace': True}}, 'Replace any existing kernel spec with this name.'), 'sys-prefix': ({'InstallKernelSpec': {'prefix': sys.prefix}}, "Install to Python's sys.prefix. Useful in conda/virtual environments."), 'debug': base_flags['debug']}

    def parse_command_line(self, argv: None | list[str]) -> None:
        """Parse the command line args."""
        super().parse_command_line(argv)
        if self.extra_args:
            self.sourcedir = self.extra_args[0]
        else:
            print('No source directory specified.', file=sys.stderr)
            self.exit(1)

    def start(self) -> None:
        """Start the application."""
        if self.user and self.prefix:
            self.exit("Can't specify both user and prefix. Please choose one or the other.")
        try:
            self.kernel_spec_manager.install_kernel_spec(self.sourcedir, kernel_name=self.kernel_name, user=self.user, prefix=self.prefix, replace=self.replace)
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if not self.user:
                    print('Perhaps you want to install with `sudo` or `--user`?', file=sys.stderr)
                self.exit(1)
            elif e.errno == errno.EEXIST:
                print(f'A kernel spec is already present at {e.filename}', file=sys.stderr)
                self.exit(1)
            raise