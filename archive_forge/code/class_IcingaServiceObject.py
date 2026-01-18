from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import url_argument_spec
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.parse import quote as urlquote
from ansible_collections.t_systems_mms.icinga_director.plugins.module_utils.icinga import (
import json
from collections import defaultdict
class IcingaServiceObject(Icinga2APIObject):
    module = None

    def __init__(self, module, path, data):
        super(IcingaServiceObject, self).__init__(module, path, data)
        self.module = module
        self.params = module.params
        self.path = path
        self.data = data
        self.object_id = None
        if 'host' in self.data:
            if self.data['host']:
                param_service_type = 'host='
                param_service_type_filter = to_text(urlquote(self.data['host']))
        if 'service_set' in self.data:
            if self.data['service_set']:
                param_service_type = 'set='
                param_service_type_filter = to_text(urlquote(self.data['service_set']))
        self.url = '/service' + '?' + 'name=' + to_text(urlquote(self.data['object_name'])) + '&' + param_service_type + param_service_type_filter

    def exists(self, find_by='name'):
        ret = super(IcingaServiceObject, self).call_url(path=self.url)
        self.object_id = to_text(urlquote(self.data['object_name']))
        if ret['code'] == 200:
            return True
        return False

    def delete(self, find_by='name'):
        return super(IcingaServiceObject, self).call_url(path=self.url, method='DELETE')

    def modify(self, find_by='name'):
        return super(IcingaServiceObject, self).call_url(path=self.url, data=self.module.jsonify(self.data), method='POST')

    def diff(self, find_by='name'):
        ret = super(IcingaServiceObject, self).call_url(path=self.url, method='GET')
        data_from_director = json.loads(self.module.jsonify(ret['data']))
        data_from_task = json.loads(self.module.jsonify(self.data))
        diff = defaultdict(dict)
        for key, value in data_from_director.items():
            value = self.scrub_diff_value(value)
            if key in data_from_task.keys() and value != data_from_task[key]:
                diff['before'][key] = '{val}'.format(val=value)
                diff['after'][key] = '{val}'.format(val=data_from_task[key])
        return diff