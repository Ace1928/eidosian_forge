import os
from distutils import log
import itertools
class DevelopInstaller(Installer):

    def _get_root(self):
        return repr(str(self.egg_path))

    def _get_target(self):
        return self.egg_link