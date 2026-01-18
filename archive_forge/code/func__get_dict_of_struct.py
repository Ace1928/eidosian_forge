from __future__ import (absolute_import, division, print_function)
import sys
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.errors import AnsibleError, AnsibleParserError
def _get_dict_of_struct(self, vm):
    """  Transform SDK Vm Struct type to Python dictionary.
             :param vm: host struct of which to create dict
             :return dict of vm struct type
        """
    vms_service = self.connection.system_service().vms_service()
    clusters_service = self.connection.system_service().clusters_service()
    vm_service = vms_service.vm_service(vm.id)
    devices = vm_service.reported_devices_service().list()
    tags = vm_service.tags_service().list()
    stats = vm_service.statistics_service().list()
    labels = vm_service.affinity_labels_service().list()
    groups = clusters_service.cluster_service(vm.cluster.id).affinity_groups_service().list()
    return {'id': vm.id, 'name': vm.name, 'host': self.connection.follow_link(vm.host).name if vm.host else None, 'cluster': self.connection.follow_link(vm.cluster).name, 'status': str(vm.status), 'description': vm.description, 'fqdn': vm.fqdn, 'os': vm.os.type, 'template': self.connection.follow_link(vm.template).name, 'creation_time': str(vm.creation_time), 'creation_time_timestamp': float(vm.creation_time.strftime('%s.%f')), 'tags': [tag.name for tag in tags], 'affinity_labels': [label.name for label in labels], 'affinity_groups': [group.name for group in groups if vm.name in [vm.name for vm in self.connection.follow_link(group.vms)]], 'statistics': dict(((stat.name, stat.values[0].datum if stat.values else None) for stat in stats)), 'devices': dict(((device.name, [ip.address for ip in device.ips]) for device in devices if device.ips))}