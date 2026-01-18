import atexit
from copy import deepcopy
import logging
import os
import shutil
import sys
from pathlib import Path
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound, PyFileConfigLoader
from IPython.core import release, crashhandler
from IPython.core.profiledir import ProfileDir, ProfileDirError
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from IPython.utils.path import ensure_dir_exists
from traitlets import (
@default('profile_dir')
def _profile_dir_default(self):
    if self._in_init_profile_dir:
        return
    self.init_profile_dir()
    return self.profile_dir