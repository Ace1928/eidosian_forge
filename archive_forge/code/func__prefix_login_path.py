from __future__ import (absolute_import, division, print_function)
import os
import os.path
import subprocess
import traceback
from ansible.errors import AnsibleError
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
def _prefix_login_path(self, remote_path):
    """ Make sure that we put files into a standard path

            If a path is relative, then we need to choose where to put it.
            ssh chooses $HOME but we aren't guaranteed that a home dir will
            exist in any given chroot.  So for now we're choosing "/" instead.
            This also happens to be the former default.

            Can revisit using $HOME instead if it's a problem
        """
    if not remote_path.startswith(os.path.sep):
        remote_path = os.path.join(os.path.sep, remote_path)
    return os.path.normpath(remote_path)