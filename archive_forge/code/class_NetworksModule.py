from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
class NetworksModule(BaseModule):

    def build_entity(self):
        if self.param('external_provider'):
            ons_service = self._connection.system_service().openstack_network_providers_service()
            on_service = ons_service.provider_service(get_id_by_name(ons_service, self.param('external_provider')))
        return otypes.Network(name=self._module.params['name'], comment=self._module.params['comment'], description=self._module.params['description'], id=self._module.params['id'], data_center=otypes.DataCenter(name=self._module.params['data_center']) if self._module.params['data_center'] else None, vlan=otypes.Vlan(self._module.params['vlan_tag'] if self._module.params['vlan_tag'] != -1 else None) if self._module.params['vlan_tag'] is not None else None, usages=[otypes.NetworkUsage.VM if self._module.params['vm_network'] else None] if self._module.params['vm_network'] is not None else None, mtu=self._module.params['mtu'], external_provider=otypes.OpenStackNetworkProvider(id=on_service.get().id) if self.param('external_provider') else None)

    def post_create(self, entity):
        self._update_label_assignments(entity)

    def _update_label_assignments(self, entity):
        if self.param('label') is None:
            return
        labels_service = self._service.service(entity.id).network_labels_service()
        labels = [lbl.id for lbl in labels_service.list()]
        if not self.param('label') in labels:
            if not self._module.check_mode:
                if labels:
                    labels_service.label_service(labels[0]).remove()
                labels_service.add(label=otypes.NetworkLabel(id=self.param('label')))
            self.changed = True

    def update_check(self, entity):
        self._update_label_assignments(entity)
        vlan_tag_changed = equal(self._module.params.get('vlan_tag'), getattr(entity.vlan, 'id', None))
        if self._module.params.get('vlan_tag') == -1:
            vlan_tag_changed = getattr(entity.vlan, 'id', None) is None
        return vlan_tag_changed and equal(self._module.params.get('comment'), entity.comment) and equal(self._module.params.get('name'), entity.name) and equal(self._module.params.get('description'), entity.description) and equal(self._module.params.get('vm_network'), True if entity.usages else False) and equal(self._module.params.get('mtu'), entity.mtu)