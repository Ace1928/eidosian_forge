import os
import os.path
import sys
from importlib import import_module, reload
from traitlets.config.configurable import Configurable
from IPython.utils.path import ensure_dir_exists
from traitlets import Instance
def _call_load_ipython_extension(self, mod):
    if hasattr(mod, 'load_ipython_extension'):
        mod.load_ipython_extension(self.shell)
        return True