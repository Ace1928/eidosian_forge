from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (
class ProxmoxDomainInfoAnsible(ProxmoxAnsible):

    def get_domain(self, realm):
        try:
            domain = self.proxmox_api.access.domains.get(realm)
        except Exception:
            self.module.fail_json(msg="Domain '%s' does not exist" % realm)
        domain['realm'] = realm
        return domain

    def get_domains(self):
        domains = self.proxmox_api.access.domains.get()
        return domains