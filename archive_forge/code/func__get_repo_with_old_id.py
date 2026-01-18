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
def _get_repo_with_old_id(self):
    """Try to get a repository with the old name."""
    repo_id = '{0}-{1}'.format(self.user, self.project)
    if repo_id in self.base.repos and '_copr' in self.base.repos[repo_id].repofile:
        file_name = self.base.repos[repo_id].repofile.split('/')[-1]
        try:
            copr_hostname = file_name.rsplit(':', 2)[0].split(':', 1)[1]
            if copr_hostname != self.host:
                return None
            return file_name
        except IndexError:
            return file_name
    return None