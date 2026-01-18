from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _match_flat_using_outdated_flatpak_format(module, binary, parsed_name, method):
    global result
    command = [binary, 'list', '--{0}'.format(method), '--app', '--columns=application']
    output = _flatpak_command(module, False, command)
    for row in output.split('\n'):
        if parsed_name.lower() == row.lower():
            return row