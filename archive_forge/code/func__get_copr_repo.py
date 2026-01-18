from __future__ import absolute_import, division, print_function
import stat
import os
import traceback
from ansible.module_utils.common import respawn
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils import distro
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
def _get_copr_repo(self):
    """Return one specific repository from all repositories on the system.

        Returns:
            The repository that a user wants to enable, disable, or remove.
        """
    repo_id = 'copr:{0}:{1}:{2}'.format(self.host, self.user, self.project)
    if repo_id not in self.base.repos:
        if self._get_repo_with_old_id() is None:
            return None
    return self.base.repos[repo_id]