import os
import sys
from itertools import product, starmap
import distutils.command.install_lib as orig
def _exclude_pkg_path(self, pkg, exclusion_path):
    """
        Given a package name and exclusion path within that package,
        compute the full exclusion path.
        """
    parts = pkg.split('.') + [exclusion_path]
    return os.path.join(self.install_dir, *parts)