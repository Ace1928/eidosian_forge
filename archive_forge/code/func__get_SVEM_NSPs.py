import os
import sys
from itertools import product, starmap
import distutils.command.install_lib as orig
def _get_SVEM_NSPs(self):
    """
        Get namespace packages (list) but only for
        single_version_externally_managed installations and empty otherwise.
        """
    if not self.distribution.namespace_packages:
        return []
    install_cmd = self.get_finalized_command('install')
    svem = install_cmd.single_version_externally_managed
    return self.distribution.namespace_packages if svem else []