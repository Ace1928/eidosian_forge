from __future__ import (absolute_import, division, print_function)
import json
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.urls import fetch_url
class Online(object):

    def __init__(self, module):
        self.module = module
        self.headers = {'Authorization': 'Bearer %s' % self.module.params.get('api_token'), 'User-Agent': self.get_user_agent_string(module), 'Content-type': 'application/json'}
        self.name = None

    def get_resources(self):
        results = self.get('/%s' % self.name)
        if not results.ok:
            raise OnlineException('Error fetching {0} ({1}) [{2}: {3}]'.format(self.name, '%s/%s' % (self.module.params.get('api_url'), self.name), results.status_code, results.json['message']))
        return results.json

    def _url_builder(self, path):
        if path[0] == '/':
            path = path[1:]
        return '%s/%s' % (self.module.params.get('api_url'), path)

    def send(self, method, path, data=None, headers=None):
        url = self._url_builder(path)
        data = self.module.jsonify(data)
        if headers is not None:
            self.headers.update(headers)
        resp, info = fetch_url(self.module, url, data=data, headers=self.headers, method=method, timeout=self.module.params.get('api_timeout'))
        if info['status'] == -1:
            self.module.fail_json(msg=info['msg'])
        return Response(resp, info)

    @staticmethod
    def get_user_agent_string(module):
        return 'ansible %s Python %s' % (module.ansible_version, sys.version.split(' ', 1)[0])

    def get(self, path, data=None, headers=None):
        return self.send('GET', path, data, headers)

    def put(self, path, data=None, headers=None):
        return self.send('PUT', path, data, headers)

    def post(self, path, data=None, headers=None):
        return self.send('POST', path, data, headers)

    def delete(self, path, data=None, headers=None):
        return self.send('DELETE', path, data, headers)

    def patch(self, path, data=None, headers=None):
        return self.send('PATCH', path, data, headers)

    def update(self, path, data=None, headers=None):
        return self.send('UPDATE', path, data, headers)