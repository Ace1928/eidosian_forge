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
def _read_all_repos(self, repo_id=None):
    """The method is used to initialize the base variable by
        repositories using the RepoReader class from dnf.

        Args:
            repo_id: Repo id of the repository we want to work with.
        """
    reader = dnf.conf.read.RepoReader(self.base.conf, None)
    for repo in reader:
        try:
            if repo_id:
                if repo.id == repo_id:
                    self.base.repos.add(repo)
                    break
            else:
                self.base.repos.add(repo)
        except dnf.exceptions.ConfigError as e:
            self.raise_exception(str(e))