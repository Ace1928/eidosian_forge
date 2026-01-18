from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
from ansible.module_utils.six.moves.urllib.parse import quote
class GrafanaOrgInterface(object):

    def __init__(self, module):
        self._module = module
        self.headers = {'Content-Type': 'application/json'}
        self.headers['Authorization'] = basic_auth_header(module.params['url_username'], module.params['url_password'])
        self.grafana_url = base.clean_url(module.params.get('url'))

    def _send_request(self, url, data=None, headers=None, method='GET'):
        if data is not None:
            data = json.dumps(data, sort_keys=True)
        if not headers:
            headers = []
        full_url = '{grafana_url}{path}'.format(grafana_url=self.grafana_url, path=url)
        resp, info = fetch_url(self._module, full_url, data=data, headers=headers, method=method)
        status_code = info['status']
        if status_code == 404:
            return None
        elif status_code == 401:
            self._module.fail_json(failed=True, msg="Unauthorized to perform action '%s' on '%s' header: %s" % (method, full_url, self.headers))
        elif status_code == 403:
            self._module.fail_json(failed=True, msg='Permission Denied')
        elif status_code == 200:
            return self._module.from_json(resp.read())
        if resp is None:
            self._module.fail_json(failed=True, msg='Cannot connect to API Grafana %s' % info['msg'], status=status_code, url=info['url'])
        else:
            self._module.fail_json(failed=True, msg='Grafana Org API answered with HTTP %d' % status_code, body=self._module.from_json(resp.read()))

    def get_actual_org(self, name):
        url = '/api/orgs/name/{name}'.format(name=quote(name))
        return self._send_request(url, headers=self.headers, method='GET')

    def create_org(self, name):
        url = '/api/orgs'
        org = dict(name=name)
        self._send_request(url, data=org, headers=self.headers, method='POST')
        return self.get_actual_org(name)

    def delete_org(self, org_id):
        url = '/api/orgs/{org_id}'.format(org_id=org_id)
        return self._send_request(url, headers=self.headers, method='DELETE')