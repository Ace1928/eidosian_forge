from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def _has_dbus_interface(self):
    """
        Checks whether subscription-manager has a D-Bus interface.

        :returns: bool -- whether subscription-manager has a D-Bus interface.
        """

    def str2int(s, default=0):
        try:
            return int(s)
        except ValueError:
            return default
    distro_id = distro.id()
    distro_version = tuple((str2int(p) for p in distro.version_parts()))
    if distro_id == 'fedora':
        return True
    return distro_version[0] == 7 and distro_version[1] >= 4 or distro_version[0] >= 8