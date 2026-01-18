import os
import shutil
import errno
from pathlib import Path
from traitlets.config.configurable import LoggingConfigurable
from ..paths import get_ipython_package_dir
from ..utils.path import expand_path, ensure_dir_exists
from traitlets import Unicode, Bool, observe
@observe('location')
def _location_changed(self, change):
    if self._location_isset:
        raise RuntimeError('Cannot set profile location more than once.')
    self._location_isset = True
    new = change['new']
    ensure_dir_exists(new)
    self.security_dir = os.path.join(new, self.security_dir_name)
    self.log_dir = os.path.join(new, self.log_dir_name)
    self.startup_dir = os.path.join(new, self.startup_dir_name)
    self.pid_dir = os.path.join(new, self.pid_dir_name)
    self.static_dir = os.path.join(new, self.static_dir_name)
    self.check_dirs()