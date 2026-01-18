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
class CheckLabExtensionsApp(BaseExtensionApp):
    description = 'Check labextension(s) by name'
    flags = check_flags
    should_check_installed_only = Bool(False, config=True, help='Whether it should check only if the extensions is installed')

    def run_task(self):
        app_options = AppOptions(app_dir=self.app_dir, logger=self.log, core_config=self.core_config, labextensions_path=self.labextensions_path)
        all_enabled = all((check_extension(arg, installed=self.should_check_installed_only, app_options=app_options) for arg in self.extra_args))
        if not all_enabled:
            self.exit(1)