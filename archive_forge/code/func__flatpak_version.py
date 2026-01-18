from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _flatpak_version(module, binary):
    global result
    command = [binary, '--version']
    output = _flatpak_command(module, False, command)
    version_number = output.split()[1]
    return version_number