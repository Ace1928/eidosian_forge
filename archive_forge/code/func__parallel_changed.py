import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
@observe('parallel')
def _parallel_changed(self, change):
    parallel_files = ['ipcontroller_config.py', 'ipengine_config.py', 'ipcluster_config.py']
    if change['new']:
        for cf in parallel_files:
            self.config_files.append(cf)
    else:
        for cf in parallel_files:
            if cf in self.config_files:
                self.config_files.remove(cf)