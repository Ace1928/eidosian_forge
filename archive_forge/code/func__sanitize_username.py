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
@staticmethod
def _sanitize_username(user):
    """Modify the group name.

        Args:
            user: User name.

        Returns:
            Modified user name if it is a group name with @.
        """
    if user[0] == '@':
        return 'group_{0}'.format(user[1:])
    return user