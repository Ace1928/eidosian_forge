from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackDiskOffering(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackDiskOffering, self).__init__(module)
        self.returns = {'disksize': 'disk_size', 'diskBytesReadRate': 'bytes_read_rate', 'diskBytesWriteRate': 'bytes_write_rate', 'diskIopsReadRate': 'iops_read_rate', 'diskIopsWriteRate': 'iops_write_rate', 'maxiops': 'iops_max', 'miniops': 'iops_min', 'hypervisorsnapshotreserve': 'hypervisor_snapshot_reserve', 'customized': 'customized', 'provisioningtype': 'provisioning_type', 'storagetype': 'storage_type', 'tags': 'storage_tags', 'displayoffering': 'display_offering'}
        self.disk_offering = None

    def get_disk_offering(self):
        args = {'name': self.module.params.get('name'), 'domainid': self.get_domain(key='id')}
        disk_offerings = self.query_api('listDiskOfferings', **args)
        if disk_offerings:
            for disk_offer in disk_offerings['diskoffering']:
                if args['name'] == disk_offer['name']:
                    self.disk_offering = disk_offer
        return self.disk_offering

    def present_disk_offering(self):
        disk_offering = self.get_disk_offering()
        if not disk_offering:
            disk_offering = self._create_offering(disk_offering)
        else:
            disk_offering = self._update_offering(disk_offering)
        return disk_offering

    def absent_disk_offering(self):
        disk_offering = self.get_disk_offering()
        if disk_offering:
            self.result['changed'] = True
            if not self.module.check_mode:
                args = {'id': disk_offering['id']}
                self.query_api('deleteDiskOffering', **args)
        return disk_offering

    def _create_offering(self, disk_offering):
        self.result['changed'] = True
        args = {'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'disksize': self.module.params.get('disk_size'), 'bytesreadrate': self.module.params.get('bytes_read_rate'), 'byteswriterate': self.module.params.get('bytes_write_rate'), 'customized': self.module.params.get('customized'), 'domainid': self.get_domain(key='id'), 'hypervisorsnapshotreserve': self.module.params.get('hypervisor_snapshot_reserve'), 'iopsreadrate': self.module.params.get('iops_read_rate'), 'iopswriterate': self.module.params.get('iops_write_rate'), 'maxiops': self.module.params.get('iops_max'), 'miniops': self.module.params.get('iops_min'), 'provisioningtype': self.module.params.get('provisioning_type'), 'diskofferingdetails': self.module.params.get('disk_offering_details'), 'storagetype': self.module.params.get('storage_type'), 'tags': self.module.params.get('storage_tags'), 'displayoffering': self.module.params.get('display_offering')}
        if not self.module.check_mode:
            res = self.query_api('createDiskOffering', **args)
            disk_offering = res['diskoffering']
        return disk_offering

    def _update_offering(self, disk_offering):
        args = {'id': disk_offering['id'], 'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name'), 'displayoffering': self.module.params.get('display_offering')}
        if self.has_changed(args, disk_offering):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateDiskOffering', **args)
                disk_offering = res['diskoffering']
        return disk_offering

    def get_result(self, resource):
        super(AnsibleCloudStackDiskOffering, self).get_result(resource)
        if resource:
            if 'tags' in resource:
                self.result['storage_tags'] = resource['tags'].split(',') or [resource['tags']]
            if 'tags' in self.result:
                del self.result['tags']
        return self.result