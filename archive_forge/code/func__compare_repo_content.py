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
def _compare_repo_content(repo_filename_path, repo_content_api):
    """Compare the contents of the stored repository with the information from the server.

        Args:
            repo_filename_path: Path to repository.
            repo_content_api: The information about the repository from the server.

        Returns:
            True, if the information matches, False otherwise.
        """
    if not os.path.isfile(repo_filename_path):
        return False
    with open(repo_filename_path, 'r') as file:
        repo_content_file = file.read()
    return repo_content_file == repo_content_api