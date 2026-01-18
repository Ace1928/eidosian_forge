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
@observe('profile')
def _profile_changed(self, change):
    self.builtin_profile_dir = os.path.join(get_ipython_package_dir(), u'config', u'profile', change['new'])