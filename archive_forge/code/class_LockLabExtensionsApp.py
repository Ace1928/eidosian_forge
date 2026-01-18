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
class LockLabExtensionsApp(BaseExtensionApp):
    description = 'Lock labextension(s) by name'
    aliases = lock_aliases
    level = Unicode('sys_prefix', help='Level at which to lock: sys_prefix, user, system').tag(config=True)

    def run_task(self):
        app_options = AppOptions(app_dir=self.app_dir, logger=self.log, core_config=self.core_config, labextensions_path=self.labextensions_path)
        [lock_extension(arg, app_options=app_options, level=self.level) for arg in self.extra_args]