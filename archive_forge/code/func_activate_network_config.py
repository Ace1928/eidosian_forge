from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def activate_network_config(client_obj, name, ignore_validation_mask):
    if utils.is_null_or_empty(name):
        return (False, False, 'Activate network config failed as name is not present.', {})
    try:
        network_resp = client_obj.network_configs.get(id=None, name=name)
        if utils.is_null_or_empty(network_resp):
            return (False, False, f"Network config '{name}' cannot be activated as it is not present.", {})
        client_obj.network_configs.activate_netconfig(id=network_resp.attrs.get('id'), ignore_validation_mask=ignore_validation_mask)
        return (True, True, f"Activated network config '{name}' successfully.", {})
    except Exception as ex:
        return (False, False, f"Activate Network config failed |'{ex}'", {})