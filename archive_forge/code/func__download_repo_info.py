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
def _download_repo_info(self):
    """Download information about the repository.

        Returns:
            Information about the repository.
        """
    distribution, version = self.short_chroot.split('-', 1)
    chroot = self.short_chroot
    while True:
        repo_info, status_code = self._get(chroot)
        if repo_info:
            return repo_info
        if distribution == 'rhel':
            chroot = 'centos-stream-8'
            distribution = 'centos'
        elif distribution == 'centos':
            if version == 'stream-8':
                version = '8'
            elif version == 'stream-9':
                version = '9'
            chroot = 'epel-{0}'.format(version)
            distribution = 'epel'
        elif str(status_code) != '404':
            self.raise_exception('This repository does not have any builds yet so you cannot enable it now.')
        else:
            self.raise_exception('Chroot {0} does not exist in {1}'.format(self.chroot, self.name))