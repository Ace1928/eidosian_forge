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
@default('labextensions_path')
def _default_labextensions_path(self):
    lab = LabApp()
    lab.load_config_file()
    return lab.extra_labextensions_path + lab.labextensions_path