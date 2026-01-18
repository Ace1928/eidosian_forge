import os
from traitlets.config.application import Application
from IPython.core.application import (
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe
class ProfileLocate(BaseIPythonApplication):
    description = 'print the path to an IPython profile dir'

    def parse_command_line(self, argv=None):
        super(ProfileLocate, self).parse_command_line(argv)
        if self.extra_args:
            self.profile = self.extra_args[0]

    def start(self):
        print(self.profile_dir.location)