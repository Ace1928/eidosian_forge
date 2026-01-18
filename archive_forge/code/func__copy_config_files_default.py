import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
def _copy_config_files_default(self):
    return True