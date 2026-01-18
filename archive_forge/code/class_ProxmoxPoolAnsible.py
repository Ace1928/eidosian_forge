from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
class ProxmoxPoolAnsible(ProxmoxAnsible):

    def is_pool_existing(self, poolid):
        """Check whether pool already exist

        :param poolid: str - name of the pool
        :return: bool - is pool exists?
        """
        try:
            pools = self.proxmox_api.pools.get()
            for pool in pools:
                if pool['poolid'] == poolid:
                    return True
            return False
        except Exception as e:
            self.module.fail_json(msg='Unable to retrieve pools: {0}'.format(e))

    def is_pool_empty(self, poolid):
        """Check whether pool has members

        :param poolid: str - name of the pool
        :return: bool - is pool empty?
        """
        return True if not self.get_pool(poolid)['members'] else False

    def create_pool(self, poolid, comment=None):
        """Create Proxmox VE pool

        :param poolid: str - name of the pool
        :param comment: str, optional - Description of a pool
        :return: None
        """
        if self.is_pool_existing(poolid):
            self.module.exit_json(changed=False, poolid=poolid, msg='Pool {0} already exists'.format(poolid))
        if self.module.check_mode:
            return
        try:
            self.proxmox_api.pools.post(poolid=poolid, comment=comment)
        except Exception as e:
            self.module.fail_json(msg='Failed to create pool with ID {0}: {1}'.format(poolid, e))

    def delete_pool(self, poolid):
        """Delete Proxmox VE pool

        :param poolid: str - name of the pool
        :return: None
        """
        if not self.is_pool_existing(poolid):
            self.module.exit_json(changed=False, poolid=poolid, msg="Pool {0} doesn't exist".format(poolid))
        if self.is_pool_empty(poolid):
            if self.module.check_mode:
                return
            try:
                self.proxmox_api.pools(poolid).delete()
            except Exception as e:
                self.module.fail_json(msg='Failed to delete pool with ID {0}: {1}'.format(poolid, e))
        else:
            self.module.fail_json(msg="Can't delete pool {0} with members. Please remove members from pool first.".format(poolid))