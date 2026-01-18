import os
import sys
from copy import copy
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Instance, List, Unicode, default
from jupyterlab.coreconfig import CoreConfig
from jupyterlab.debuglog import DebugLogFileMixin
from .commands import (
from .federated_labextensions import build_labextension, develop_labextension_py, watch_labextension
from .labapp import LabApp
class UninstallLabExtensionApp(BaseExtensionApp):
    description = 'Uninstall labextension(s) by name'
    flags = uninstall_flags
    all = Bool(False, config=True, help='Whether to uninstall all extensions')

    def run_task(self):
        self.deprecation_warning('Uninstalling extensions with the jupyter labextension uninstall command is now deprecated and will be removed in a future major version of JupyterLab.')
        self.extra_args = self.extra_args or [os.getcwd()]
        options = AppOptions(app_dir=self.app_dir, logger=self.log, labextensions_path=self.labextensions_path, core_config=self.core_config)
        return any((uninstall_extension(arg, all_=self.all, app_options=options) for arg in self.extra_args))