from __future__ import absolute_import, division, print_function
import abc
import os
import stat
import traceback
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
@_restore_all_on_failure
def _safe_secure_move(self, sources_and_destinations):
    """Moves a list of files from 'source' to 'destination' and restores 'destination' from backup upon failure.
           If 'destination' does not already exist, then 'source' permissions are preserved to prevent
           exposing protected data ('atomic_move' uses the 'destination' base directory mask for
           permissions if 'destination' does not already exists).
        """
    for source, destination in sources_and_destinations:
        if os.path.exists(destination):
            self.module.atomic_move(source, destination)
        else:
            self.module.preserved_copy(source, destination)