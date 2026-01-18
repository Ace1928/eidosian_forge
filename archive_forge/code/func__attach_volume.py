from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import xrange
from ansible.module_utils.common.text.converters import to_native
def _attach_volume(module, profitbricks, datacenter, volume):
    """
    Attaches a volume.

    This will attach a volume to the server.

    module : AnsibleModule object
    profitbricks: authenticated profitbricks object.

    Returns:
        True if the volume was attached, false otherwise
    """
    server = module.params.get('server')
    if server:
        if not uuid_match.match(server):
            server_list = profitbricks.list_servers(datacenter)
            for s in server_list['items']:
                if server == s['properties']['name']:
                    server = s['id']
                    break
        try:
            return profitbricks.attach_volume(datacenter, server, volume)
        except Exception as e:
            module.fail_json(msg='failed to attach volume: %s' % to_native(e), exception=traceback.format_exc())