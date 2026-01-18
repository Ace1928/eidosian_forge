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
class KernelSpecApp(Application):
    """An app to manage kernel specs."""
    version = __version__
    name = 'jupyter kernelspec'
    description = 'Manage Jupyter kernel specifications.'
    subcommands = Dict({'list': (ListKernelSpecs, ListKernelSpecs.description.splitlines()[0]), 'install': (InstallKernelSpec, InstallKernelSpec.description.splitlines()[0]), 'uninstall': (RemoveKernelSpec, 'Alias for remove'), 'remove': (RemoveKernelSpec, RemoveKernelSpec.description.splitlines()[0]), 'install-self': (InstallNativeKernelSpec, InstallNativeKernelSpec.description.splitlines()[0]), 'provisioners': (ListProvisioners, ListProvisioners.description.splitlines()[0])})
    aliases = {}
    flags = {}

    def start(self) -> None:
        """Start the application."""
        if self.subapp is None:
            print('No subcommand specified. Must specify one of: %s' % list(self.subcommands))
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()