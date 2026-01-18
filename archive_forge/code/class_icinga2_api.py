from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, url_argument_spec
class icinga2_api:
    module = None

    def __init__(self, module):
        self.module = module

    def call_url(self, path, data='', method='GET'):
        headers = {'Accept': 'application/json', 'X-HTTP-Method-Override': method}
        url = self.module.params.get('url') + '/' + path
        rsp, info = fetch_url(module=self.module, url=url, data=data, headers=headers, method=method, use_proxy=self.module.params['use_proxy'])
        body = ''
        if rsp:
            body = json.loads(rsp.read())
        if info['status'] >= 400:
            body = info['body']
        return {'code': info['status'], 'data': body}

    def check_connection(self):
        ret = self.call_url('v1/status')
        if ret['code'] == 200:
            return True
        return False

    def exists(self, hostname):
        data = {'filter': 'match("' + hostname + '", host.name)'}
        ret = self.call_url(path='v1/objects/hosts', data=self.module.jsonify(data))
        if ret['code'] == 200:
            if len(ret['data']['results']) == 1:
                return True
        return False

    def create(self, hostname, data):
        ret = self.call_url(path='v1/objects/hosts/' + hostname, data=self.module.jsonify(data), method='PUT')
        return ret

    def delete(self, hostname):
        data = {'cascade': 1}
        ret = self.call_url(path='v1/objects/hosts/' + hostname, data=self.module.jsonify(data), method='DELETE')
        return ret

    def modify(self, hostname, data):
        ret = self.call_url(path='v1/objects/hosts/' + hostname, data=self.module.jsonify(data), method='POST')
        return ret

    def diff(self, hostname, data):
        ret = self.call_url(path='v1/objects/hosts/' + hostname, method='GET')
        changed = False
        ic_data = ret['data']['results'][0]
        for key in data['attrs']:
            if key not in ic_data['attrs'].keys():
                changed = True
            elif data['attrs'][key] != ic_data['attrs'][key]:
                changed = True
        return changed