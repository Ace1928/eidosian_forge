from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ngine_io.cloudstack.plugins.module_utils.cloudstack import (
class InstanceRdnsRecord(AnsibleCloudStack):

    def __init__(self, module):
        super(InstanceRdnsRecord, self).__init__(module)
        self.name = self.module.params.get('name')
        self.content = self.module.params.get('content')
        self.returns = {'domain': 'domain'}
        self.instance = None

    def get_instance(self):
        instance = self.instance
        if not instance:
            args = {'fetch_list': True}
            instances = self.query_api('listVirtualMachines', **args)
            if instances:
                for v in instances:
                    if self.name.lower() in [v['name'].lower(), v['displayname'].lower(), v['id']]:
                        self.instance = v
                        break
        return self.instance

    def get_record(self, instance):
        result = {}
        res = self.query_api('queryReverseDnsForVirtualMachine', id=instance['id'])
        nics = res['virtualmachine'].get('nic', [])
        defaultnics = [nic for nic in nics if nic.get('isdefault', False)]
        if len(defaultnics) > 0:
            domains = [record['domainname'] for record in defaultnics[0].get('reversedns', []) if 'domainname' in record]
            if len(domains) > 0:
                result = {'domainname': domains[0]}
        return result

    def present_record(self):
        instance = self.get_instance()
        if not instance:
            self.module.fail_json(msg='No compute instance with name=%s found. ' % self.name)
        data = {'domainname': self.content}
        record = self.get_record(instance)
        if self.has_changed(data, record):
            self.result['changed'] = True
            self.result['diff']['before'] = record
            self.result['diff']['after'] = data
            if not self.module.check_mode:
                self.query_api('updateReverseDnsForVirtualMachine', id=instance['id'], domainname=data['domainname'])
        return data

    def absent_record(self):
        instance = self.get_instance()
        if instance:
            record = self.get_record(instance)
            if record:
                self.result['diff']['before'] = record
                self.result['changed'] = True
                if not self.module.check_mode:
                    self.query_api('deleteReverseDnsFromVirtualMachine', id=instance['id'])
            return record

    def get_result(self, resource):
        self.result['instance_rdns_domain'] = resource
        return self.result