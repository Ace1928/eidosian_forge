from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class ProxmoxAnsible(object):
    """Base class for Proxmox modules"""

    def __init__(self, module):
        if not HAS_PROXMOXER:
            module.fail_json(msg=missing_required_lib('proxmoxer'), exception=PROXMOXER_IMP_ERR)
        self.module = module
        self.proxmoxer_version = proxmoxer_version
        self.proxmox_api = self._connect()
        try:
            self.proxmox_api.version.get()
        except Exception as e:
            module.fail_json(msg='%s' % e, exception=traceback.format_exc())

    def _connect(self):
        api_host = self.module.params['api_host']
        api_user = self.module.params['api_user']
        api_password = self.module.params['api_password']
        api_token_id = self.module.params['api_token_id']
        api_token_secret = self.module.params['api_token_secret']
        validate_certs = self.module.params['validate_certs']
        auth_args = {'user': api_user}
        if api_password:
            auth_args['password'] = api_password
        else:
            if self.proxmoxer_version < LooseVersion('1.1.0'):
                self.module.fail_json('Using "token_name" and "token_value" require proxmoxer>=1.1.0')
            auth_args['token_name'] = api_token_id
            auth_args['token_value'] = api_token_secret
        try:
            return ProxmoxAPI(api_host, verify_ssl=validate_certs, **auth_args)
        except Exception as e:
            self.module.fail_json(msg='%s' % e, exception=traceback.format_exc())

    def version(self):
        try:
            apiversion = self.proxmox_api.version.get()
            return LooseVersion(apiversion['version'])
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve Proxmox VE version: %s' % e)

    def get_node(self, node):
        try:
            nodes = [n for n in self.proxmox_api.nodes.get() if n['node'] == node]
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve Proxmox VE node: %s' % e)
        return nodes[0] if nodes else None

    def get_nextvmid(self):
        try:
            return self.proxmox_api.cluster.nextid.get()
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve next free vmid: %s' % e)

    def get_vmid(self, name, ignore_missing=False, choose_first_if_multiple=False):
        try:
            vms = [vm['vmid'] for vm in self.proxmox_api.cluster.resources.get(type='vm') if vm.get('name') == name]
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve list of VMs filtered by name %s: %s' % (name, e))
        if not vms:
            if ignore_missing:
                return None
            self.module.fail_json(msg='No VM with name %s found' % name)
        elif len(vms) > 1:
            self.module.fail_json(msg='Multiple VMs with name %s found, provide vmid instead' % name)
        return vms[0]

    def get_vm(self, vmid, ignore_missing=False):
        try:
            vms = [vm for vm in self.proxmox_api.cluster.resources.get(type='vm') if vm['vmid'] == int(vmid)]
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve list of VMs filtered by vmid %s: %s' % (vmid, e))
        if vms:
            return vms[0]
        else:
            if ignore_missing:
                return None
            self.module.fail_json(msg='VM with vmid %s does not exist in cluster' % vmid)

    def api_task_ok(self, node, taskid):
        try:
            status = self.proxmox_api.nodes(node).tasks(taskid).status.get()
            return status['status'] == 'stopped' and status['exitstatus'] == 'OK'
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve API task ID from node %s: %s' % (node, e))

    def get_pool(self, poolid):
        """Retrieve pool information

        :param poolid: str - name of the pool
        :return: dict - pool information
        """
        try:
            return self.proxmox_api.pools(poolid).get()
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve pool %s information: %s' % (poolid, e))

    def get_storages(self, type):
        """Retrieve storages information

        :param type: str, optional - type of storages
        :return: list of dicts - array of storages
        """
        try:
            return self.proxmox_api.storage.get(type=type)
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve storages information with type %s: %s' % (type, e))

    def get_storage_content(self, node, storage, content=None, vmid=None):
        try:
            return self.proxmox_api.nodes(node).storage(storage).content().get(content=content, vmid=vmid)
        except Exception as e:
            self.module.fail_json(msg='Unable to list content on %s, %s for %s and %s: %s' % (node, storage, content, vmid, e))