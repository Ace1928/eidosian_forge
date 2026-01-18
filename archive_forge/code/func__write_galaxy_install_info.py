from __future__ import (absolute_import, division, print_function)
import errno
import datetime
import functools
import os
import tarfile
import tempfile
from collections.abc import MutableSequence
from shutil import rmtree
from ansible import context
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.galaxy.api import GalaxyAPI
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.yaml import yaml_dump, yaml_load
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import open_url
from ansible.playbook.role.requirement import RoleRequirement
from ansible.utils.display import Display
from ansible.utils.path import is_subpath, unfrackpath
def _write_galaxy_install_info(self):
    """
        Writes a YAML-formatted file to the role's meta/ directory
        (named .galaxy_install_info) which contains some information
        we can use later for commands like 'list' and 'info'.
        """
    info = dict(version=self.version, install_date=datetime.datetime.now(datetime.timezone.utc).strftime('%c'))
    if not os.path.exists(os.path.join(self.path, 'meta')):
        os.makedirs(os.path.join(self.path, 'meta'))
    info_path = os.path.join(self.path, self.META_INSTALL)
    with open(info_path, 'w+') as f:
        try:
            self._install_info = yaml_dump(info, f)
        except Exception:
            return False
    return True