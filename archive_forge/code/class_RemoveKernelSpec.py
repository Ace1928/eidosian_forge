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
class RemoveKernelSpec(JupyterApp):
    """An app to remove a kernel spec."""
    version = __version__
    description = 'Remove one or more Jupyter kernelspecs by name.'
    examples = 'jupyter kernelspec remove python2 [my_kernel ...]'
    force = Bool(False, config=True, help="Force removal, don't prompt for confirmation.")
    spec_names = List(Unicode())
    kernel_spec_manager = Instance(KernelSpecManager)

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(data_dir=self.data_dir, parent=self)
    flags = {'f': ({'RemoveKernelSpec': {'force': True}}, force.help)}
    flags.update(JupyterApp.flags)

    def parse_command_line(self, argv: list[str] | None) -> None:
        """Parse the command line args."""
        super().parse_command_line(argv)
        if self.extra_args:
            self.spec_names = sorted(set(self.extra_args))
        else:
            self.exit('No kernelspec specified.')

    def start(self) -> None:
        """Start the application."""
        self.kernel_spec_manager.ensure_native_kernel = False
        spec_paths = self.kernel_spec_manager.find_kernel_specs()
        missing = set(self.spec_names).difference(set(spec_paths))
        if missing:
            self.exit("Couldn't find kernel spec(s): %s" % ', '.join(missing))
        if not (self.force or self.answer_yes):
            print('Kernel specs to remove:')
            for name in self.spec_names:
                path = spec_paths.get(name, name)
                print(f'  {name.ljust(20)}\t{path.ljust(20)}')
            answer = input('Remove %i kernel specs [y/N]: ' % len(self.spec_names))
            if not answer.lower().startswith('y'):
                return
        for kernel_name in self.spec_names:
            try:
                path = self.kernel_spec_manager.remove_kernel_spec(kernel_name)
            except OSError as e:
                if e.errno == errno.EACCES:
                    print(e, file=sys.stderr)
                    print('Perhaps you want sudo?', file=sys.stderr)
                    self.exit(1)
                else:
                    raise
            print(f'Removed {path}')