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
class UpdateLabExtensionApp(BaseExtensionApp):
    description = 'Update labextension(s)'
    flags = update_flags
    all = Bool(False, config=True, help='Whether to update all extensions')

    def run_task(self):
        self.deprecation_warning('Updating extensions with the jupyter labextension update command is now deprecated and will be removed in a future major version of JupyterLab.')
        if not self.all and (not self.extra_args):
            self.log.warning('Specify an extension to update, or use --all to update all extensions')
            return False
        app_options = AppOptions(app_dir=self.app_dir, logger=self.log, core_config=self.core_config, labextensions_path=self.labextensions_path)
        if self.all:
            return update_extension(all_=True, app_options=app_options)
        return any((update_extension(name=arg, app_options=app_options) for arg in self.extra_args))