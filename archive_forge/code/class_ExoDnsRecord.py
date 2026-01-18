from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.exoscale import (ExoDns, exo_dns_argument_spec,
class ExoDnsRecord(ExoDns):

    def __init__(self, module):
        super(ExoDnsRecord, self).__init__(module)
        self.domain = self.module.params.get('domain').lower()
        self.name = self.module.params.get('name').lower()
        if self.name == self.domain:
            self.name = ''
        self.multiple = self.module.params.get('multiple')
        self.record_type = self.module.params.get('record_type')
        self.content = self.module.params.get('content')

    def _create_record(self, record):
        self.result['changed'] = True
        data = {'record': {'name': self.name, 'record_type': self.record_type, 'content': self.content, 'ttl': self.module.params.get('ttl'), 'prio': self.module.params.get('prio')}}
        self.result['diff']['after'] = data['record']
        if not self.module.check_mode:
            record = self.api_query('/domains/%s/records' % self.domain, 'POST', data)
        return record

    def _update_record(self, record):
        data = {'record': {'name': self.name, 'content': self.content, 'ttl': self.module.params.get('ttl'), 'prio': self.module.params.get('prio')}}
        if self.has_changed(data['record'], record['record']):
            self.result['changed'] = True
            if not self.module.check_mode:
                record = self.api_query('/domains/%s/records/%s' % (self.domain, record['record']['id']), 'PUT', data)
        return record

    def get_record(self):
        domain = self.module.params.get('domain')
        records = self.api_query('/domains/%s/records' % domain, 'GET')
        result = {}
        for r in records:
            if r['record']['record_type'] != self.record_type:
                continue
            r_name = r['record']['name'].lower()
            r_content = r['record']['content']
            if r_name == self.name:
                if not self.multiple:
                    if result:
                        self.module.fail_json(msg='More than one record with record_type=%s and name=%s params. Use multiple=yes for more than one record.' % (self.record_type, self.name))
                    else:
                        result = r
                elif r_content == self.content:
                    return r
        return result

    def present_record(self):
        record = self.get_record()
        if not record:
            record = self._create_record(record)
        else:
            record = self._update_record(record)
        return record

    def absent_record(self):
        record = self.get_record()
        if record:
            self.result['diff']['before'] = record
            self.result['changed'] = True
            if not self.module.check_mode:
                self.api_query('/domains/%s/records/%s' % (self.domain, record['record']['id']), 'DELETE')
        return record

    def get_result(self, resource):
        if resource:
            self.result['exo_dns_record'] = resource['record']
            self.result['exo_dns_record']['domain'] = self.domain
        return self.result